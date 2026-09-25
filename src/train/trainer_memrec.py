"""
MemRec Trainer v2 (evaluation only, no training)
Supports eval_feedback modes: gt, random, none
Supports parallel evaluation for speedup
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from tqdm import tqdm
import json
import random
import time
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.models import LLMClient, MemRecAgent
from src.models.llm_client import RequestBudget, DurableRequestBudget
from src.train.books_run_journal import BooksRunJournal
from src.data import RecDataset

# Thread-safe random number generator
_thread_local = threading.local()


class MemRecTrainer:
    """MemRec Trainer (evaluation only, reranking)"""
    
    def __init__(
        self,
        model,  # None for MemRec
        dataset: RecDataset,
        config: Dict,
        device: torch.device
    ):
        """
        Initialize trainer
        
        Args:
            model: None (MemRec doesn't need it)
            dataset: RecDataset (needs item_metadata loaded)
            config: Configuration dictionary
            device: Device (for API consistency)
        """
        self.dataset = dataset
        self.config = config
        self.device = device
        self.save_conversations = config.get('save_llm_conversations', False)
        self.conversation_file = config.get('conversation_file', None)
        
        # Load instructions if available (like iAgent)
        if dataset.instructions is None and hasattr(dataset, 'load_instructions'):
            try:
                dataset.load_instructions()
                if dataset.instructions:
                    print(f"  ✓ Loaded instructions for {len(dataset.instructions)} users")
            except Exception as e:
                print(f"  ⚠ Could not load instructions: {e}")
        
        # Extract configuration
        self.n_eval_candidates = config.get('n_eval_candidates', 10)
        self.use_pregenerated_candidates = config.get('use_pregenerated_candidates', False)
        self.n_eval_users = config.get('n_eval_users', None)
        self.eval_user_list = config.get('eval_user_list', None)  # Path to JSON file with fixed user list
        self.eval_cohort = config.get('eval_cohort', None)  # dev | heldout | all, Books protocol
        if self.eval_cohort and self.eval_user_list:
            raise ValueError('Set either eval_cohort or eval_user_list, not both')
        self.topk = config.get('topk', [1, 3, 5])
        self.metrics = config.get('metrics', ['Hit', 'NDCG'])
        
        # MemRec configuration
        memrec_config = config.get('memrec', {})
        self.k = memrec_config.get('k', 16)
        self.tau = memrec_config.get('tau_tokens', memrec_config.get('tau', 1800))
        self.n_facets = memrec_config.get('n_facets', 7)
        self.temperature = memrec_config.get('temperature', 0.0)
        self.max_tokens = memrec_config.get('max_tokens', 1000)
        self.mix_min_users = memrec_config.get('mix_min_users', 4)
        self.mix_min_items = memrec_config.get('mix_min_items', 6)
        
        # Ablation study controls
        self.enable_stage_r = memrec_config.get('enable_stage_r', True)  # Enable Stage-R (memory retrieval)
        self.enable_stage_w = memrec_config.get('enable_stage_w', True)  # Enable Stage-W (memory writing)
        
        # Reranker configuration (v2 new)
        self.reranker_mode = memrec_config.get('reranker_mode', 'vector')  # 'vector' or 'llm'
        
        # Pruner configuration (v2 new)
        pruner_config = memrec_config.get('pruner', {})
        self.pruner_mode = pruner_config.get('mode', 'hybrid_rule')  # 'hybrid_rule' or 'learned_mlp'
        self.pruner_checkpoint = pruner_config.get('checkpoint', None)
        
        # Write configuration
        write_config = memrec_config.get('write', {})
        self.fanout_cap = write_config.get('fanout_cap', 8)
        
        # eval_feedback mode
        self.eval_feedback = config.get('eval_feedback', 'none')  # gt, random, none
        
        # Warm-up configuration (v2 new)
        warmup_config = config.get('warmup', {})
        self.warmup_enabled = warmup_config.get('enabled', False)
        self.warmup_rounds = warmup_config.get('rounds', 1)
        self.warmup_user_scope = config.get('warmup_user_scope', 'eval')
        self._books_run_journal = None
        if self.warmup_user_scope not in ('eval', 'all'):
            raise ValueError('warmup_user_scope must be eval or all')
        
        # Debug settings
        self.debug = config.get('debug', False)
        self.debug_log_file = config.get('debug_log_file', None)
        
        # LLM configuration
        provider_config = config.get('provider', {})
        provider_name = provider_config.get('name', 'azure_openai')  # Default to azure_openai
        self.llm_model = provider_config.get('model', config.get('llm_model', 'gpt-4o-mini'))
        self.llm_revision = provider_config.get('revision')
        self.api_endpoint = provider_config.get('endpoint', config.get('api_endpoint'))
        self.api_key = provider_config.get('api_key', config.get('api_key'))
        # Allow provider to override api_version
        self.api_version = provider_config.get('api_version', config.get('api_version', '2024-08-01-preview'))
        
        # Initialize LLM client (for Stage-R and Stage-W)
        self.llm_client = LLMClient(
            api_endpoint=self.api_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            model=self.llm_model,
            provider_name=provider_name,
            save_conversations=self.save_conversations,
            conversation_log_path=self.conversation_file,
            sdk_max_retries=int(provider_config.get('sdk_max_retries', 2)),
        )
        
        # If reranker_mode is 'llm', create a separate client with the same
        # provider credentials by default. This matters for Azure: passing only
        # the root-level endpoint/key would otherwise make Stage-ReRank fall back
        # to a different environment configuration than Stage-R/Stage-W.
        if self.reranker_mode == "llm":
            reranker_llm_model = provider_config.get(
                'reranker_model', config.get('llm_model', self.llm_model)
            )
            reranker_provider_name = provider_config.get('reranker_name', provider_name)
            reranker_endpoint = provider_config.get('reranker_endpoint', self.api_endpoint)
            reranker_api_key = provider_config.get('reranker_api_key', self.api_key)
            reranker_api_version = provider_config.get('reranker_api_version', self.api_version)
            
            self.reranker_llm_client = LLMClient(
                api_endpoint=reranker_endpoint,
                api_key=reranker_api_key,
                api_version=reranker_api_version,
                model=reranker_llm_model,
                provider_name=reranker_provider_name,
                save_conversations=self.save_conversations,
                conversation_log_path=self.conversation_file,
                sdk_max_retries=int(provider_config.get('sdk_max_retries', 2)),
            )
            
            print(f"\nLLM Client Configuration:")
            print(f"  Stage-R & Stage-W: {self.llm_model} @ {self.api_endpoint}")
            print(f"  Stage-ReRank: {reranker_llm_model} @ {reranker_endpoint}")
        else:
            self.reranker_llm_client = None

        if self.use_pregenerated_candidates:
            if not self.llm_revision:
                raise ValueError('Fixed-list benchmark requires a pinned provider.revision')
            if self.reranker_llm_client and self.reranker_llm_client.model != self.llm_model:
                raise ValueError('Full MemRec benchmark requires the same LM for Stage-R/W and Stage-ReRank')
        
        # Load item metadata
        print("\nLoading item metadata for MemRec...")
        self.dataset.load_item_metadata()
        if self.use_pregenerated_candidates:
            self.dataset.load_ranked_lists()
            if not self.dataset.ranked_lists:
                raise ValueError('Pre-generated candidates requested, but none were loaded')
            # Fail before any LLM call if the fixed benchmark list is corrupt.
            for user_id, target_item in self.dataset.test_data.items():
                candidates = self._evaluation_candidates(user_id, target_item, 'test')
                seen_history = set(self.dataset.get_user_all_items(user_id)) - {target_item}
                if seen_history.intersection(candidates):
                    raise ValueError(f'Pre-generated candidates overlap user history: {user_id}')
            if self.dataset.data_path.stem == 'instructrec-books':
                from src.data.books_protocol import BOOKS_CANDIDATE_SHA256, candidate_digest
                observed_digest = candidate_digest(self.dataset.ranked_lists, self.dataset.test_data)
                if observed_digest != BOOKS_CANDIDATE_SHA256:
                    raise ValueError('Books candidate manifest hash differs from locked protocol')
        
        # Initialize MemRec Agent (v2: three stages)
        print("\nInitializing MemRec Agent v2...")
        self.agent = MemRecAgent(
            dataset=self.dataset,
            llm_client=self.llm_client,
            k=self.k,
            tau=self.tau,
            n_facets=self.n_facets,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            mix_min_users=self.mix_min_users,
            mix_min_items=self.mix_min_items,
            fanout_cap=self.fanout_cap,
            reranker_mode=self.reranker_mode,
            pruner_mode=self.pruner_mode,
            pruner_checkpoint=self.pruner_checkpoint,
            enable_stage_r=self.enable_stage_r,  # Ablation: control Stage-R
            # True "no memory system at all" baseline: only when warmup never ran (no M_u/M_collab
            # ever exists) *and* Stage-R is off. "w/o Collab. Read" keeps warmup on, so it stays
            # out of vanilla_mode and just loses the (now-conditional) facets section instead.
            vanilla_mode=(not self.warmup_enabled and not self.enable_stage_r),
            upstream_empty_facets_prompt=memrec_config.get('upstream_empty_facets_prompt', False),
            reranker_llm_client=self.reranker_llm_client,  # Separate LLMClient for reranker
            debug=self.debug
        )
        
        print(f"\nInitialized MemRec Trainer v2")
        print(f"  LLM Model: {self.llm_model}")
        print(f"  Pruner: mode={self.pruner_mode}, k={self.k} (mix: ≥{self.mix_min_users} users, ≥{self.mix_min_items} items)")
        if self.pruner_checkpoint:
            print(f"    Checkpoint: {self.pruner_checkpoint}")
        print(f"  Reranker: {self.reranker_mode}")
        print(f"  Token budget τ: {self.tau}")
        print(f"  N facets: {self.n_facets}")
        print(f"  Evaluation candidates: {self.n_eval_candidates}")
        print(f"  Eval feedback mode: {self.eval_feedback}")
        if self.warmup_enabled:
            print(f"  🔥 Training: ENABLED")
        if self.n_eval_users:
            print(f"  Eval users (sampled): {self.n_eval_users}")
        if self.debug:
            print(f"  🔍 Debug mode: ENABLED")
    
    def _init_debug_logger(self, save_dir):
        """Initialize debug log file"""
        if not self.debug:
            return
        
        if self.debug_log_file:
            log_file = Path(self.debug_log_file)
        else:
            log_file = Path(save_dir) / "debug.txt" if save_dir else Path("debug_memrec.txt")
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.debug_log_file = str(log_file)
        
        # Clear existing file
        with open(self.debug_log_file, 'w', encoding='utf-8') as f:
            f.write(f"MemRec Debug Log\n")
            f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")
        
        print(f"  🔍 Debug log: {self.debug_log_file}")
    
    def _log_debug(self, message, level="INFO"):
        """Write to debug log"""
        if not self.debug or not self.debug_log_file:
            return
        
        timestamp = time.strftime('%H:%M:%S')
        log_line = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.debug_log_file, 'a', encoding='utf-8') as f:
            f.write(log_line)
    
    def _log_user_detail(self, user_id, target_item, candidates, details, ranked_items, pos):
        """Log detailed information for a single user"""
        if not self.debug:
            return
        
        self._log_debug("\n" + "="*100)
        self._log_debug(f"USER {user_id}")
        self._log_debug("="*100)
        
        # Basic information
        self._log_debug(f"\nTarget item: {target_item}")
        if target_item in self.dataset.item_metadata:
            meta = self.dataset.item_metadata[target_item]
            self._log_debug(f"  Title: {meta.get('title', 'N/A')[:100]}")
        
        self._log_debug(f"\nCandidates ({len(candidates)} items): {candidates}")
        
        # Pruned subgraph
        if 'pruned_subgraph' in details:
            ps = details['pruned_subgraph']
            self._log_debug(f"\nPruned neighbors: {len(ps['neighbors'])} ({ps['n_items']} items, {ps['n_users']} users)")
            self._log_debug("Selected neighbors:")
            for i, n in enumerate(ps['neighbors'], 1):
                self._log_debug(f"  [{i}] {n['type'].upper()}-{n['id']}: score={n['score']:.4f}")
        
        # Packed context
        if 'packed_context' in details:
            cb = details['packed_context']
            self._log_debug(f"\nContext: {cb['n_neighbors']} neighbors, ~{cb['estimated_tokens']} tokens")
        
        # Facets
        self._log_debug(f"\nFacets ({len(details['facets'])} generated):")
        for i, facet in enumerate(details['facets'], 1):
            self._log_debug(f"  [{i}] {facet.get('facet', facet.get('text', 'N/A'))}")
            self._log_debug(f"      Confidence: {facet.get('confidence', 0):.2f}")
        
        # Evidence
        evidence = details.get('evidence', [])
        if evidence:
            self._log_debug(f"\nEvidence ({len(evidence)} entries):")
            for i, ev in enumerate(evidence[:10], 1):  # Only show first 10
                self._log_debug(f"  [{i}] Facet {ev['facet_idx']} ← {ev['neighbor_id']} (weight={ev['weight']:.2f})")
        
        # Rerank scores (v2: replaces candidate_notes)
        rerank_scores = details.get('rerank_scores', [])
        if rerank_scores:
            self._log_debug(f"\nCandidate scoring ({len(rerank_scores)} items):")
            for i, score_entry in enumerate(rerank_scores, 1):
                cid = score_entry['item_id']
                score = score_entry['score']
                rationale = score_entry.get('rationale', 'N/A')
                marker = " ⭐ TARGET" if cid == target_item else ""
                self._log_debug(f"  [{i}] Item {cid}: score={score:.3f}{marker}")
                self._log_debug(f"      {rationale[:100]}")
        
        # Ranking result
        self._log_debug(f"\nRanking:")
        self._log_debug(f"  Original: {candidates}")
        self._log_debug(f"  Reranked: {ranked_items}")
        self._log_debug(f"  Target position: {pos+1}/{len(candidates)}")
        
        if pos < 1:
            self._log_debug("  ✓ Hit@1!")
        elif pos < 3:
            self._log_debug("  ✓ Hit@3!")
        elif pos < 5:
            self._log_debug("  ✓ Hit@5!")
    
    def _evaluate_single_user(
        self, 
        user_id: int, 
        target_item: int, 
        target_data: Dict,
        split: str = 'test',
        lock: threading.Lock = None
    ) -> Tuple[int, Dict]:
        """
        Evaluate single user (for parallel processing)
        
        Returns:
            (position, timings_dict)
        """
        if self.use_pregenerated_candidates:
            candidates = self._evaluation_candidates(user_id, target_item, split)
        else:
            # Legacy sampler retained for exploratory runs only.
            if not hasattr(_thread_local, 'rng'):
                _thread_local.rng = np.random.RandomState(seed=hash(threading.current_thread().ident) % (2**32))
            all_items = set(range(self.dataset.n_items))
            user_history = set(self.dataset.get_user_train_items(user_id))
            negative_pool = list(all_items - user_history - {target_item})
            if len(negative_pool) < self.n_eval_candidates - 1:
                return (-1, {})
            negative_items = _thread_local.rng.choice(negative_pool, size=self.n_eval_candidates - 1, replace=False).tolist()
            candidates = [target_item] + negative_items
            _thread_local.rng.shuffle(candidates)
        
        # Get instruction if available
        instruction = None
        if self.dataset.instructions and user_id in self.dataset.instructions:
            instruction = self.dataset.instructions[user_id].get('instruction')
        
        # Rerank
        try:
            ranked_items, details = self.agent.rerank(
                user_id=user_id,
                candidates=candidates,
                instruction=instruction,  # Pass instruction to reranker
                return_details=True,
                debug_logger=None  # No debug output in parallel mode
            )
            
            # Extract timings
            timings = details.get('timings', {})
            stage_r_time = timings.get('stage_r', 0.0)
            stage_rr_time = timings.get('stage_rr', 0.0)
            
            # Stage-W (only if enabled)
            stage_w_time = 0.0
            patches_applied = 0
            if self.enable_stage_w and self.eval_feedback != 'none':
                feedback = self._generate_feedback(
                    user_id=user_id,
                    target_item=target_item,
                    ranked_items=ranked_items,
                    mode=self.eval_feedback
                )
                
                if feedback:
                    t0 = time.time()
                    write_result = self.agent.write(
                        user_id=user_id,
                        feedback=feedback,
                        recent_facets=details.get('facets', []),
                        pruned_subgraph=details.get('pruned_subgraph', None),
                        debug_logger=None
                    )
                    stage_w_time = time.time() - t0
                    patches_applied = write_result.get('stats', {}).get('total_applied', 0)
            
            # Calculate ranking position
            try:
                position = ranked_items.index(target_item)
            except ValueError:
                position = len(ranked_items)
            
            return (position, {
                'stage_r_time': stage_r_time,
                'stage_rr_time': stage_rr_time,
                'stage_w_time': stage_w_time,
                'patches_applied': patches_applied
            })
            
        except Exception as e:
            if lock:
                with lock:
                    print(f"\nError evaluating user {user_id}: {e}")
            else:
                print(f"\nError evaluating user {user_id}: {e}")
            return (-1, {})
    
    def evaluate(self, split: str = 'test', save_dir: str = None, parallel: bool = False, n_workers: int = 16) -> Dict:
        """
        Evaluate MemRec reranking performance
        
        Args:
            split: 'test' or 'valid'
            save_dir: Save directory (optional)
            parallel: Whether to use parallel evaluation
            n_workers: Number of parallel workers
            
        Returns:
            Metrics dictionary
        """
        # Initialize debug logger (disabled in parallel mode)
        if self.use_pregenerated_candidates and split != 'test':
            raise ValueError('Original InstructRec ranked_lists are test-only; validation needs separate candidates')
        if self.use_pregenerated_candidates and parallel:
            raise ValueError('Fixed-list benchmark requires serial memory reads/writes until snapshot isolation is implemented')
        if self.use_pregenerated_candidates and self.eval_feedback != 'none':
            raise ValueError('Fixed-list benchmark forbids test-time feedback writes')
        if self.debug and save_dir and not parallel:
            self._init_debug_logger(save_dir)
        # Get evaluation data
        if split == 'test':
            target_data = self.dataset.test_data
        else:
            target_data = self.dataset.valid_data
        
        # Sample users (if specified)
        # Priority: eval_user_list > n_eval_users > all users
        if self.eval_cohort:
            if split != 'test' or self.dataset.data_path.stem != 'instructrec-books':
                raise ValueError('eval_cohort is defined only for InstructRec Books test rows')
            from src.data.books_protocol import books_cohorts
            cohorts, manifest = books_cohorts(list(target_data.keys()))
            if self.eval_cohort not in cohorts:
                raise ValueError(f'Unknown Books eval cohort: {self.eval_cohort}')
            eval_user_ids = cohorts[self.eval_cohort]
            if self.n_eval_users is not None:
                eval_user_ids = eval_user_ids[:self.n_eval_users]
            print(f"Books cohort: {self.eval_cohort}; {len(eval_user_ids)} users")
            print(f"Cohort SHA-256: {manifest['cohort_sha256'][self.eval_cohort]}")
        elif self.eval_user_list:
            # Load fixed user list from JSON file
            user_list_path = Path(self.eval_user_list)
            if not user_list_path.is_absolute():
                # Relative to project root
                user_list_path = Path(__file__).parent.parent.parent / user_list_path
            
            print(f"\n📋 Loading fixed user list from: {user_list_path}")
            with open(user_list_path, 'r') as f:
                user_data = json.load(f)
            
            loaded_user_ids = user_data['user_ids']
            # Filter to only include users in target_data
            eval_user_ids = [uid for uid in loaded_user_ids if uid in target_data]
            print(f"  ✓ Loaded {len(loaded_user_ids)} user IDs, {len(eval_user_ids)} are in {split} set")
            
            if len(eval_user_ids) < len(loaded_user_ids):
                print(f"  ⚠ {len(loaded_user_ids) - len(eval_user_ids)} users from list not found in {split} set")
        elif self.n_eval_users and self.n_eval_users < len(target_data):
            all_user_ids = list(target_data.keys())
            random.seed(self.config.get('seed', 42))
            eval_user_ids = random.sample(all_user_ids, self.n_eval_users)
            print(f"\nSampled {self.n_eval_users} users for evaluation")
        else:
            eval_user_ids = list(target_data.keys())

        if self.use_pregenerated_candidates:
            if self.eval_cohort == 'heldout' and self.n_eval_users is not None:
                raise ValueError('Held-out benchmark must score the entire locked cohort')
            if self.warmup_user_scope == 'eval' and len(eval_user_ids) > 30:
                raise ValueError('Partial warm-up is permitted only for a 20–30-user smoke')
            if self.warmup_user_scope == 'eval' and len(eval_user_ids) < 20:
                raise ValueError('Fixed-list smoke must contain 20–30 users')
            if self.warmup_user_scope == 'all' and self.eval_cohort == 'heldout' and len(eval_user_ids) != 5377:
                raise ValueError('Held-out cohort size differs from the locked 5,377-user protocol')

        # ========== Training Phase ==========
        # Simulate memory construction and propagation before testing
        warmup_user_ids = (
            sorted(target_data) if self.warmup_user_scope == 'all' else eval_user_ids
        )
        print(f'Warm-up scope: {self.warmup_user_scope}; {len(warmup_user_ids)} users')
        if self.use_pregenerated_candidates:
            if getattr(self.llm_client, 'sdk_max_retries', 0) != 0 or (
                self.reranker_llm_client
                and getattr(self.reranker_llm_client, 'sdk_max_retries', 0) != 0
            ):
                raise ValueError('Fixed-list benchmark requires sdk_max_retries=0 for physical request accounting')
            expected_warmup_calls = (
                len(warmup_user_ids) * self.warmup_rounds
                * (int(self.enable_stage_r) + int(self.reranker_mode == 'llm') + int(self.enable_stage_w))
                if self.warmup_enabled else 0
            )
            expected_eval_calls = len(eval_user_ids) * (
                int(self.enable_stage_r) + int(self.reranker_mode == 'llm')
            )
            expected_calls = expected_warmup_calls + expected_eval_calls
            retry_fraction = float(self.config.get('llm_retry_reserve_fraction', 0.10))
            if not 0 <= retry_fraction <= 1:
                raise ValueError('llm_retry_reserve_fraction must be between 0 and 1')
            hard_cap = expected_calls + max(10, math.ceil(expected_calls * retry_fraction))
            journal_path = os.getenv('MEMREC_BOOKS_RUN_JOURNAL_DB')
            budget_path = os.getenv('MEMREC_BOOKS_REQUEST_BUDGET_DB')
            if bool(journal_path) != bool(budget_path):
                raise ValueError('Books full-run journal and durable budget must be enabled together')
            if journal_path:
                if (self.warmup_user_scope != 'all' or self.eval_cohort != 'dev'
                        or parallel):
                    raise ValueError('Resumable Books journal is only for serial full-dev evaluation')
                run_id = os.getenv('MEMREC_BOOKS_FULL_RUN_ID')
                commit = os.getenv('MEMREC_BOOKS_FULL_GIT_COMMIT')
                if not run_id or not commit:
                    raise ValueError('Resumable Books journal needs a pinned run ID and git commit')
                contract = {
                    'schema_version': 1, 'run_id': run_id, 'git_commit': commit,
                    'model_revision': self.llm_revision, 'config': self.config,
                    'warmup_user_ids': warmup_user_ids, 'eval_user_ids': eval_user_ids,
                }
                self._books_run_journal = BooksRunJournal(journal_path, contract)
                self.llm_budget = DurableRequestBudget(
                    hard_cap, budget_path, self._books_run_journal.contract_sha256
                )
            else:
                self.llm_budget = RequestBudget(hard_cap)
            self.llm_client.request_budget = self.llm_budget
            self.llm_client.fail_fast = True
            self.llm_client.strict_schema_validation = True
            if self.reranker_llm_client:
                self.reranker_llm_client.request_budget = self.llm_budget
                self.reranker_llm_client.fail_fast = True
                self.reranker_llm_client.strict_schema_validation = True
            print(f'LLM physical-request contract: expected={expected_calls}, hard_cap={hard_cap}')
        before_warmup_calls = (
            self.agent.n_stage_r_calls,
            self.agent.n_stage_rr_calls,
            self.agent.n_stage_w_calls,
        )
        self._run_training(warmup_user_ids, target_data, split, parallel=parallel, n_workers=n_workers)
        warmup_calls = (
            self.agent.n_stage_r_calls - before_warmup_calls[0],
            self.agent.n_stage_rr_calls - before_warmup_calls[1],
            self.agent.n_stage_w_calls - before_warmup_calls[2],
        )
        
        # Evaluation loop
        ranking_positions = []  # Store target item ranking positions
        prediction_records = []  # Fixed-list benchmark: one record per evaluated user
        n_success = 0
        
        # Stage statistics (v2: three stages)
        stage_r_times = []
        stage_rr_times = []  # Rerank
        stage_w_times = []
        stage_w_applied = []
        
        # Token usage tracking per stage
        # Record initial token statistics at evaluation start
        initial_llm_stats = self.llm_client.get_token_stats()
        initial_rr_llm_stats = self.reranker_llm_client.get_token_stats() if self.reranker_llm_client else None
        
        # For accumulating token usage per stage
        stage_r_input_tokens = []
        stage_r_output_tokens = []
        stage_rr_input_tokens = []
        stage_rr_output_tokens = []
        stage_w_input_tokens = []
        stage_w_output_tokens = []

        eval_records = []
        if self._books_run_journal is not None:
            eval_records = self._books_run_journal.read_phase('eval', eval_user_ids)
            for record in eval_records:
                ranking_positions.append(record['position'])
                prediction_records.append(record['prediction'])
                n_success += bool(record['success'])
                stage_r_times.append(record['stage_r_time'])
                stage_rr_times.append(record['stage_rr_time'])
                stage_r_input_tokens.append(record['stage_r_input'])
                stage_r_output_tokens.append(record['stage_r_output'])
                stage_rr_input_tokens.append(record['stage_rr_input'])
                stage_rr_output_tokens.append(record['stage_rr_output'])
            if eval_records:
                self._restore_journal_client_state(eval_records[-1]['client_state'])
                print(f'Replayed {len(eval_records)}/{len(eval_user_ids)} committed dev predictions')
        
        print(f"\nEvaluating MemRec on {split} set ({len(eval_user_ids)} users)...")
        print(f"Eval feedback mode: {self.eval_feedback}")
        if parallel:
            print(f"🚀 Parallel mode: {n_workers} workers")
        
        # ========== Parallel Mode ==========
        if parallel:
            lock = threading.Lock()
            futures = []
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # Submit all tasks
                for user_id in eval_user_ids:
                    target_item = target_data[user_id]
                    future = executor.submit(
                        self._evaluate_single_user,
                        user_id, target_item, target_data, split, lock
                    )
                    futures.append(future)
                
                # Collect results
                with tqdm(total=len(futures), desc=f"MemRec {split}") as pbar:
                    for future in as_completed(futures):
                        pos, timings = future.result()
                        if pos >= 0:  # Valid result
                            ranking_positions.append(pos)
                            n_success += 1
                            stage_r_times.append(timings.get('stage_r_time', 0.0))
                            stage_rr_times.append(timings.get('stage_rr_time', 0.0))
                            stage_w_times.append(timings.get('stage_w_time', 0.0))
                            stage_w_applied.append(timings.get('patches_applied', 0))
                        pbar.update(1)
            
            # In parallel mode, cannot record token statistics per user separately
            # Calculate overall token usage after evaluation ends, then divide by number of users to get average
            final_llm_stats = self.llm_client.get_token_stats()
            final_rr_llm_stats = self.reranker_llm_client.get_token_stats() if self.reranker_llm_client else None
            
            # Calculate overall token usage increment
            total_stage_r_input = final_llm_stats['total_input_tokens'] - initial_llm_stats['total_input_tokens']
            total_stage_r_output = final_llm_stats['total_output_tokens'] - initial_llm_stats['total_output_tokens']
            
            if self.reranker_mode == "llm" and final_rr_llm_stats:
                total_stage_rr_input = final_rr_llm_stats['total_input_tokens'] - initial_rr_llm_stats['total_input_tokens']
                total_stage_rr_output = final_rr_llm_stats['total_output_tokens'] - initial_rr_llm_stats['total_output_tokens']
            else:
                total_stage_rr_input = 0
                total_stage_rr_output = 0
            
            # Calculate averages (Note: assumes all users executed Stage-R and Stage-ReRank)
            n_users_with_rerank = len(stage_r_times)
            if n_users_with_rerank > 0:
                # Note: In parallel mode, Stage-R and Stage-ReRank token usage may include other operations
                # So averages may not be precise, but can serve as reference
                avg_stage_r_input = total_stage_r_input / n_users_with_rerank
                avg_stage_r_output = total_stage_r_output / n_users_with_rerank
                avg_stage_rr_input = total_stage_rr_input / n_users_with_rerank if self.reranker_mode == "llm" else 0
                avg_stage_rr_output = total_stage_rr_output / n_users_with_rerank if self.reranker_mode == "llm" else 0
                
                stage_r_input_tokens = [avg_stage_r_input] * n_users_with_rerank
                stage_r_output_tokens = [avg_stage_r_output] * n_users_with_rerank
                stage_rr_input_tokens = [avg_stage_rr_input] * n_users_with_rerank
                stage_rr_output_tokens = [avg_stage_rr_output] * n_users_with_rerank
            
            # Stage-W token usage needs separate calculation (not all users execute Stage-W)
            if len(stage_w_times) > 0:
                # In parallel mode, cannot precisely calculate Stage-W token usage
                # Use approximate value: assume Stage-W token usage similar to Stage-R
                # Or can set to 0 to indicate cannot precisely count
                stage_w_input_tokens = [0] * len(stage_w_times)  # Cannot precisely count
                stage_w_output_tokens = [0] * len(stage_w_times)  # Cannot precisely count
        
        # ========== Serial Mode (original logic, debug support retained) ==========
        else:
            for eval_seq, user_id in enumerate(tqdm(eval_user_ids, desc=f"MemRec {split}")):
                if eval_seq < len(eval_records):
                    continue
                target_item = target_data[user_id]
                
                if self.use_pregenerated_candidates:
                    candidates = self._evaluation_candidates(user_id, target_item, split)
                else:
                    # Legacy sampler retained for exploratory runs only.
                    all_items = set(range(self.dataset.n_items))
                    user_history = set(self.dataset.get_user_train_items(user_id))
                    negative_pool = list(all_items - user_history - {target_item})
                    if len(negative_pool) < self.n_eval_candidates - 1:
                        print(f"Warning: Not enough negative samples for user {user_id}")
                        continue
                    negative_items = random.sample(negative_pool, self.n_eval_candidates - 1)
                    candidates = [target_item] + negative_items
                    random.shuffle(candidates)
                
                # Stage-R: Reranking
                try:
                    if self.debug:
                        self._log_debug(f"\n{'='*100}")
                        self._log_debug(f"Processing user {user_id}")
                        self._log_debug(f"{'='*100}")
                    
                    # Get instruction if available
                    instruction = None
                    if self.dataset.instructions and user_id in self.dataset.instructions:
                        instruction = self.dataset.instructions[user_id].get('instruction')
                        if self.debug and instruction:
                            self._log_debug(f"\n💬 User Instruction: {instruction[:100]}...")
                    
                    # Record token statistics before rerank
                    before_rerank_llm_stats = self.llm_client.get_token_stats()
                    before_rerank_rr_llm_stats = self.reranker_llm_client.get_token_stats() if self.reranker_llm_client else None
                    
                    t0 = time.time()
                    ranked_items, details = self.agent.rerank(
                        user_id=user_id,
                        candidates=candidates,
                        instruction=instruction,  # Pass instruction to reranker
                        return_details=True,  # Need details for Stage-W
                        debug_logger=self._log_debug if self.debug else None
                    )
                    total_time = time.time() - t0
                    
                    # Record token statistics after rerank
                    after_rerank_llm_stats = self.llm_client.get_token_stats()
                    after_rerank_rr_llm_stats = self.reranker_llm_client.get_token_stats() if self.reranker_llm_client else None
                    
                    # Calculate token usage
                    # Note: Stage-R uses self.llm_client, Stage-ReRank uses self.reranker_llm_client (if reranker_mode is 'llm')
                    if self.reranker_mode == "llm" and self.reranker_llm_client:
                        # Stage-R and Stage-ReRank use different clients, can distinguish
                        # Stage-R uses llm_client
                        stage_r_input = after_rerank_llm_stats['total_input_tokens'] - before_rerank_llm_stats['total_input_tokens']
                        stage_r_output = after_rerank_llm_stats['total_output_tokens'] - before_rerank_llm_stats['total_output_tokens']
                        # Stage-ReRank uses reranker_llm_client
                        stage_rr_input = after_rerank_rr_llm_stats['total_input_tokens'] - before_rerank_rr_llm_stats['total_input_tokens']
                        stage_rr_output = after_rerank_rr_llm_stats['total_output_tokens'] - before_rerank_rr_llm_stats['total_output_tokens']
                    else:
                        # Vector mode: only Stage-R uses llm_client, Stage-ReRank doesn't use LLM
                        stage_r_input = after_rerank_llm_stats['total_input_tokens'] - before_rerank_llm_stats['total_input_tokens']
                        stage_r_output = after_rerank_llm_stats['total_output_tokens'] - before_rerank_llm_stats['total_output_tokens']
                        stage_rr_input = 0
                        stage_rr_output = 0
                    
                    # v2: Extract stage-by-stage timings
                    timings = details.get('timings', {})
                    stage_r_time = timings.get('stage_r', 0.0)
                    stage_rr_time = timings.get('stage_rr', 0.0)
                    stage_r_times.append(stage_r_time)
                    stage_rr_times.append(stage_rr_time)
                    
                    # Record token usage
                    stage_r_input_tokens.append(stage_r_input)
                    stage_r_output_tokens.append(stage_r_output)
                    stage_rr_input_tokens.append(stage_rr_input)
                    stage_rr_output_tokens.append(stage_rr_output)
                    
                    if self.debug:
                        self._log_debug(f"\nStage-R: {stage_r_time*1000:.1f}ms")
                        self._log_debug(f"Stage-ReRank ({self.reranker_mode}): {stage_rr_time*1000:.1f}ms")
                        self._log_debug(f"Total rerank time: {total_time*1000:.1f}ms")
                    
                    # Fixed-list benchmark treats malformed/fallback output as
                    # a miss, never as an accidentally correct ranking.
                    if self.use_pregenerated_candidates:
                        pos, failure = self._benchmark_position(
                            target_item, candidates, ranked_items, details
                        )
                        ranking_positions.append(pos)
                        n_success += failure is None
                        prediction_records.append({
                            'user_id': user_id,
                            'target_item': target_item,
                            'candidates': candidates,
                            'ranked_items': ranked_items,
                            'target_position': pos,
                            'failure': failure,
                        })
                        if self.debug and failure is None:
                            self._log_user_detail(user_id, target_item, candidates, details, ranked_items, pos)
                    elif target_item in ranked_items:
                        pos = ranked_items.index(target_item)
                        ranking_positions.append(pos)
                        n_success += 1
                        if self.debug:
                            self._log_user_detail(user_id, target_item, candidates, details, ranked_items, pos)
                    else:
                        ranking_positions.append(len(candidates))
                        if self.debug:
                            self._log_debug(f"WARNING: Target item {target_item} not in ranked_items for user {user_id}", level="WARNING")
                    
                    # Stage-W: Write (based on eval_feedback mode)
                    try:
                        if self.eval_feedback != 'none':
                            feedback = self._generate_feedback(
                                user_id=user_id,
                                target_item=target_item,
                                ranked_items=ranked_items,
                                mode=self.eval_feedback
                            )
                            
                            if feedback:
                                if self.debug:
                                    self._log_debug(f"\nStage-W: feedback={feedback}")
                                
                                # Record token statistics before Stage-W
                                before_stage_w_stats = self.llm_client.get_token_stats()
                                
                                t0 = time.time()
                                write_result = self.agent.write(
                                    user_id=user_id,
                                    feedback=feedback,
                                    recent_facets=details.get('facets', []),
                                    pruned_subgraph=details.get('pruned_subgraph', None),
                                    debug_logger=self._log_debug if self.debug else None
                                )
                                stage_w_time = time.time() - t0
                                
                                # Record token statistics after Stage-W
                                after_stage_w_stats = self.llm_client.get_token_stats()
                                
                                # Calculate Stage-W token usage
                                stage_w_input = after_stage_w_stats['total_input_tokens'] - before_stage_w_stats['total_input_tokens']
                                stage_w_output = after_stage_w_stats['total_output_tokens'] - before_stage_w_stats['total_output_tokens']
                                
                                stage_w_times.append(stage_w_time)
                                stage_w_applied.append(write_result['stats']['total_applied'])
                                stage_w_input_tokens.append(stage_w_input)
                                stage_w_output_tokens.append(stage_w_output)
                                
                                if self.debug:
                                    self._log_debug(f"Stage-W completed in {stage_w_time*1000:.1f}ms")
                                    self._log_debug(f"Applied patches: {write_result['stats']}")
                    except Exception as e_write:
                        # Stage-W errors don't affect evaluation results
                        if self.debug:
                            self._log_debug(f"WARNING: Stage-W failed for user {user_id}: {e_write}", level="WARNING")
                    
                except Exception as e:
                    if self.use_pregenerated_candidates:
                        raise
                    # Entire evaluation flow errors
                    print(f"\nError evaluating user {user_id}: {e}")
                    ranking_positions.append(len(candidates))
                    if self.use_pregenerated_candidates:
                        prediction_records.append({
                            'user_id': user_id,
                            'target_item': target_item,
                            'candidates': candidates,
                            'ranked_items': [],
                            'target_position': len(candidates),
                            'failure': type(e).__name__,
                        })
                    
                    if self.debug:
                        import traceback
                        self._log_debug(f"ERROR for user {user_id}: {e}", level="ERROR")
                        self._log_debug(traceback.format_exc(), level="ERROR")
                
                if self._books_run_journal is not None:
                    if (len(prediction_records) != eval_seq + 1
                            or len(stage_r_times) != eval_seq + 1):
                        raise RuntimeError('Books evaluation journal cannot commit incomplete user')
                    record = prediction_records[-1]
                    self._books_run_journal.append('eval', eval_seq, user_id, {
                        'position': ranking_positions[-1], 'prediction': record,
                        'success': record['failure'] is None,
                        'stage_r_time': stage_r_times[-1],
                        'stage_rr_time': stage_rr_times[-1],
                        'stage_r_input': stage_r_input_tokens[-1],
                        'stage_r_output': stage_r_output_tokens[-1],
                        'stage_rr_input': stage_rr_input_tokens[-1],
                        'stage_rr_output': stage_rr_output_tokens[-1],
                        'client_state': self._journal_client_state(),
                    })
                continue
        
        if self.use_pregenerated_candidates and len(ranking_positions) != len(eval_user_ids):
            raise RuntimeError('Benchmark dropped evaluation users; metric denominator would be wrong')
        if self.use_pregenerated_candidates and len(prediction_records) != len(eval_user_ids):
            raise RuntimeError('Benchmark did not record every evaluated user')
        if self.use_pregenerated_candidates and save_dir:
            output = Path(save_dir) / f'{split}_predictions.jsonl'
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('w', encoding='utf-8') as handle:
                for record in prediction_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f'Per-user predictions saved to {output}')

        # Calculate metrics
        metrics = self._compute_metrics(ranking_positions, len(eval_user_ids))
        metrics['n_eval_users'] = len(eval_user_ids)
        metrics['n_warmup_users'] = len(warmup_user_ids) if self.warmup_enabled else 0
        metrics['warmup_user_scope'] = self.warmup_user_scope
        metrics['n_rankings'] = len(ranking_positions)
        metrics['n_failed_rankings'] = len(eval_user_ids) - n_success
        metrics['llm_model_revision'] = self.llm_revision
        if self.use_pregenerated_candidates:
            metrics['llm_physical_requests'] = self.llm_budget.used
            metrics['llm_request_hard_cap'] = self.llm_budget.limit
        metrics['n_stage_r_warmup_calls'] = warmup_calls[0]
        metrics['n_stage_rr_warmup_calls'] = warmup_calls[1]
        metrics['n_stage_w_warmup_calls'] = warmup_calls[2]
        
        # Add Stage statistics (v2: three stages)
        metrics['n_stage_r_calls'] = len(stage_r_times)
        metrics['avg_stage_r_time_ms'] = np.mean(stage_r_times) * 1000 if stage_r_times else 0.0
        metrics['n_stage_rr_calls'] = len(stage_rr_times)
        metrics['avg_stage_rr_time_ms'] = np.mean(stage_rr_times) * 1000 if stage_rr_times else 0.0
        metrics['reranker_mode'] = self.reranker_mode
        metrics['pruner_mode'] = self.pruner_mode
        metrics['n_stage_w_calls'] = len(stage_w_times)
        metrics['avg_stage_w_time_ms'] = np.mean(stage_w_times) * 1000 if stage_w_times else 0.0
        metrics['avg_patches_applied'] = np.mean(stage_w_applied) if stage_w_applied else 0.0
        
        # Print results
        print(f"\n{'='*80}")
        print(f"MemRec v2 Evaluation Results ({split} set)")
        print(f"{'='*80}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        
        print(f"\nStage timings (avg):")
        print(f"  Stage-R: {metrics['avg_stage_r_time_ms']:.1f} ms")
        print(f"  Stage-ReRank ({self.reranker_mode}): {metrics['avg_stage_rr_time_ms']:.1f} ms")
        if metrics['n_stage_w_calls'] > 0:
            print(f"  Stage-W: {metrics['avg_stage_w_time_ms']:.1f} ms")
        
        # Stage Token Usage statistics
        if len(stage_r_input_tokens) > 0:
            avg_stage_r_input = np.mean(stage_r_input_tokens)
            avg_stage_r_output = np.mean(stage_r_output_tokens)
            avg_stage_rr_input = np.mean(stage_rr_input_tokens) if len(stage_rr_input_tokens) > 0 else 0
            avg_stage_rr_output = np.mean(stage_rr_output_tokens) if len(stage_rr_output_tokens) > 0 else 0
            avg_stage_w_input = np.mean(stage_w_input_tokens) if len(stage_w_input_tokens) > 0 else 0
            avg_stage_w_output = np.mean(stage_w_output_tokens) if len(stage_w_output_tokens) > 0 else 0
            
            print(f"\nStage Token Usage:")
            print(f"  Stage-R:  Input ~{avg_stage_r_input:.0f} | Output ~{avg_stage_r_output:.0f}")
            if self.reranker_mode == "llm":
                print(f"  Stage-RR: Input ~{avg_stage_rr_input:.0f} | Output ~{avg_stage_rr_output:.0f}")
            else:
                print(f"  Stage-RR: Input ~0 | Output ~0 (vector mode)")
            if metrics['n_stage_w_calls'] > 0:
                print(f"  Stage-W:  Input ~{avg_stage_w_input:.0f} | Output ~{avg_stage_w_output:.0f}")
        
        # Token usage statistics (overall)
        token_stats = self.llm_client.get_token_stats()
        if token_stats['total_requests'] > 0:
            metrics['llm_token_stats'] = token_stats
            print(f"\nLLM Token Usage Statistics:")
            print(f"  Total requests: {token_stats['total_requests']}")
            print(f"  Total input tokens: {token_stats['total_input_tokens']}")
            print(f"  Total output tokens: {token_stats['total_output_tokens']}")
            print(f"  Total tokens: {token_stats['total_tokens']}")
            print(f"  Avg input tokens per request: {token_stats['avg_input_tokens']:.1f}")
            print(f"  Avg output tokens per request: {token_stats['avg_output_tokens']:.1f}")
        
        # If reranker_llm_client is used, also show its statistics
        if self.reranker_llm_client:
            rr_token_stats = self.reranker_llm_client.get_token_stats()
            if rr_token_stats['total_requests'] > 0:
                print(f"\nReranker LLM Token Usage Statistics:")
                print(f"  Total requests: {rr_token_stats['total_requests']}")
                print(f"  Total input tokens: {rr_token_stats['total_input_tokens']}")
                print(f"  Total output tokens: {rr_token_stats['total_output_tokens']}")
                print(f"  Total tokens: {rr_token_stats['total_tokens']}")
                print(f"  Avg input tokens per request: {rr_token_stats['avg_input_tokens']:.1f}")
                print(f"  Avg output tokens per request: {rr_token_stats['avg_output_tokens']:.1f}")
        
        # Storage statistics
        storage_stats = self.agent.get_stats()
        print(f"\nStorage statistics:")
        for k, v in storage_stats['storage'].items():
            print(f"  {k}: {v}")
        
        # Debug summary
        if self.debug and self.debug_log_file:
            self._log_debug("\n" + "="*100)
            self._log_debug("EVALUATION SUMMARY")
            self._log_debug("="*100)
            for k, v in metrics.items():
                self._log_debug(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            self._log_debug(f"Storage: {storage_stats['storage']}")
            if token_stats['total_requests'] > 0:
                self._log_debug(f"Token Usage: {token_stats}")
            print(f"\n✓ Debug log saved to: {self.debug_log_file}")
        
        return metrics

    @staticmethod
    def _benchmark_position(target_item, candidates, ranked_items, details):
        """Validate a complete candidate permutation before scoring its target."""
        if len(ranked_items) != len(candidates) or set(ranked_items) != set(candidates):
            return len(candidates), 'malformed_ranking'
        scores = details.get('rerank_scores', [])
        if scores and all(str(row.get('rationale', '')).lower() == 'error' for row in scores):
            return len(candidates), 'llm_fallback'
        if scores and any(
            not isinstance(row.get('score'), (int, float))
            or isinstance(row.get('score'), bool)
            or not math.isfinite(row['score'])
            or not 0 <= row['score'] <= 1
            for row in scores
        ):
            return len(candidates), 'invalid_score'
        return ranked_items.index(target_item), None

    def _evaluation_candidates(self, user_id: int, target_item: int, split: str) -> list[int]:
        """Return the original, fixed InstructRec test candidates without reshuffling."""
        if split != 'test':
            raise ValueError('Pre-generated ranked_lists are test-only')
        if not self.dataset.ranked_lists or user_id not in self.dataset.ranked_lists:
            raise ValueError(f'Missing pre-generated candidates for user {user_id}')
        candidates = [int(item) for item in self.dataset.ranked_lists[user_id]]
        if (len(candidates) != self.n_eval_candidates
                or len(set(candidates)) != len(candidates)
                or target_item not in candidates
                or any(item < 0 or item >= self.dataset.n_items for item in candidates)):
            raise ValueError(f'Invalid pre-generated candidates for user {user_id}')
        return candidates
    
    def _construct_candidates(
        self,
        user_id: int,
        target_item: int,
        history: list[int]
    ) -> list[int]:
        """
        Construct candidate set: target + negatives
        
        Args:
            user_id: User ID
            target_item: Target item
            history: User interaction history
            
        Returns:
            Candidate item list (shuffled)
        """
        if self.use_pregenerated_candidates:
            # Warm-up needs its own candidates: the original ranked_lists are
            # test-only. Stable user/target RNG keeps both paired arms aligned
            # without materializing ~7,000 full-catalog negative arrays.
            rng = random.Random(
                int(self.config.get('seed', 42)) * 1_000_003
                + int(user_id) * 100_003 + int(target_item)
            )
            forbidden = set(self.dataset.get_user_all_items(user_id))
            forbidden.update(history)
            forbidden.add(target_item)
            if self.dataset.n_items - len(forbidden) < self.n_eval_candidates - 1:
                raise ValueError(f'Not enough warm-up negatives for user {user_id}')
            negatives = set()
            while len(negatives) < self.n_eval_candidates - 1:
                item = rng.randrange(self.dataset.n_items)
                if item not in forbidden:
                    negatives.add(item)
            candidates = [target_item] + sorted(negatives)
            rng.shuffle(candidates)
            return candidates

        # Legacy warm-up sampler for exploratory runs.
        user_positives = set(history + [target_item])
        negative_pool = [item for item in self.dataset.user_negatives.get(user_id, []) 
                         if item not in user_positives]
        
        if len(negative_pool) < self.n_eval_candidates - 1:
            # Not enough negative samples, use all available negatives
            negative_items = negative_pool
        else:
            # Random sampling
            negative_items = random.sample(negative_pool, self.n_eval_candidates - 1)
        
        # Construct candidate set and shuffle
        candidates = [target_item] + negative_items
        random.shuffle(candidates)
        
        return candidates
    
    def _warmup_single_user(self, user_id, warmup_round, target_offset, split, lock=None):
        """
        Execute training for a single user (for parallel processing)
        
        Returns:
            bool: Whether memory was successfully updated
        """
        # Get user's complete interaction history
        all_interactions = self.dataset.get_user_history(user_id, split)
        
        # Check if history is long enough
        if len(all_interactions) <= target_offset:
            return False
        
        # Calculate history and target
        target_idx = -(target_offset + 1)
        history = all_interactions[:target_idx]
        target_item = all_interactions[target_idx]
        
        # Get instruction
        instruction = None
        if hasattr(self, 'instructions') and self.instructions:
            instruction = self.instructions.get(user_id, None)
        
        # Construct candidate set
        candidates = self._construct_candidates(
            user_id=user_id,
            target_item=target_item,
            history=history
        )
        
        try:
            # Three-stage recommendation
            ranked_items, details = self.agent.rerank(
                user_id=user_id,
                candidates=candidates,
                instruction=instruction,
                debug_logger=None  # Parallel mode doesn't support debug
            )
            
            # Check target position
            if target_item not in ranked_items:
                return False
            
            target_pos = ranked_items.index(target_item)
            
            # Generate feedback and update memory
            if target_pos < 10:  # top-10 triggers Stage-W
                feedback = {
                    'action': 'CLICK',
                    'item_id': target_item,
                    'position': target_pos
                }
                
                # Stage-W: Write memory
                write_result = self.agent.write(
                    user_id=user_id,
                    feedback=feedback,
                    recent_facets=details.get('facets', []),
                    pruned_subgraph=details.get('pruned_subgraph', None),
                    debug_logger=None  # Parallel mode doesn't support debug
                )
                
                return True
            
            return False
            
        except Exception as e:
            if self.use_pregenerated_candidates:
                raise
            return False

    def _journal_client_state(self):
        fields = ('total_input_tokens', 'total_output_tokens', 'total_requests',
                  'total_physical_requests', 'total_cache_hits')
        return [
            {field: getattr(client, field, 0) for field in fields} if client else None
            for client in (self.llm_client, self.reranker_llm_client)
        ]

    def _restore_journal_client_state(self, states):
        for client, state in zip((self.llm_client, self.reranker_llm_client), states):
            if client and state:
                for field, value in state.items():
                    setattr(client, field, value)

    def _run_training_with_journal(self, user_ids: list[int], split: str):
        """Replay committed memory writes, then continue serial Stage-R/RR/W."""
        journal = self._books_run_journal
        expected = user_ids * self.warmup_rounds
        committed = journal.read_phase('warmup', expected)
        for record in committed:
            self.agent.storage.apply_mutation_record(record['memory'])
        if committed:
            last = committed[-1]
            self.agent.n_stage_r_calls, self.agent.n_stage_rr_calls, self.agent.n_stage_w_calls = last['stage_calls']
            self._restore_journal_client_state(last['client_state'])
            print(f'Replayed {len(committed)}/{len(expected)} committed warm-up users from journal')

        for warmup_round in range(self.warmup_rounds):
            start = warmup_round * len(user_ids)
            end = start + len(user_ids)
            n_updates = sum(bool(row['did_write']) for row in committed[start:end])
            target_offset = self.warmup_rounds - warmup_round
            for offset, user_id in enumerate(tqdm(user_ids, desc='MemRec Training')):
                seq = start + offset
                if seq < len(committed):
                    continue
                self.agent.storage.begin_mutation_record()
                did_write = self._warmup_single_user(
                    user_id, warmup_round, target_offset, split
                )
                memory = self.agent.storage.finish_mutation_record()
                journal.append('warmup', seq, user_id, {
                    'did_write': bool(did_write), 'memory': memory,
                    'stage_calls': [self.agent.n_stage_r_calls,
                                    self.agent.n_stage_rr_calls,
                                    self.agent.n_stage_w_calls],
                    'client_state': self._journal_client_state(),
                })
                n_updates += bool(did_write)
            print(f'  ✓ {n_updates}/{len(user_ids)} users updated memories')
        print('\n' + '=' * 80)
        print('🔥 MemRec Training Complete!')
    
    def _run_training(
        self,
        eval_user_ids: list[int],
        target_data: dict,
        split: str = 'test',
        parallel: bool = False,
        n_workers: int = 16
    ):
        """
        Execute Training phase: simulate memory construction using sliding window
        
        Args:
            eval_user_ids: List of evaluation user IDs
            target_data: Test data (user_id -> target_item)
            split: Dataset split ('test' or 'valid')
            parallel: Whether to use parallel mode
            n_workers: Number of parallel workers
        """
        if not self.warmup_enabled:
            return
        
        print(f"\n{'='*80}")
        print(f"🔥 Training MemRec")
        print(f"{'='*80}")
        if parallel:
            print(f"🚀 Parallel mode: {n_workers} workers")

        if getattr(self, '_books_run_journal', None) is not None:
            self._run_training_with_journal(eval_user_ids, split)
            return
        
        # Execute training for each user
        for warmup_round in range(self.warmup_rounds):
            # Calculate target position for current round
            target_offset = self.warmup_rounds - warmup_round
            
            n_updates = 0  # Count how many users updated memory in this round
            
            # ========== Parallel Mode ==========
            if parallel:
                lock = threading.Lock()
                futures = []
                
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    # Submit all tasks
                    for user_id in eval_user_ids:
                        future = executor.submit(
                            self._warmup_single_user,
                            user_id, warmup_round, target_offset, split, lock
                        )
                        futures.append(future)
                    
                    # Collect results
                    with tqdm(total=len(futures), desc=f"MemRec Training") as pbar:
                        for future in as_completed(futures):
                            success = future.result()
                            if success:
                                n_updates += 1
                            pbar.update(1)
            
            # ========== Serial Mode (debug support retained) ==========
            else:
                for user_id in tqdm(eval_user_ids, desc=f"MemRec Training"):
                    # Get user's complete interaction history
                    all_interactions = self.dataset.get_user_history(user_id, split)
                    
                    # Check if history is long enough
                    # target_offset=1 requires at least 2 items (T-1 + history)
                    if len(all_interactions) <= target_offset:
                        continue  # Not enough history, skip this user
                    
                    # Calculate history and target
                    # target_offset=1 => T-1 (2nd from end) => idx=-2
                    # target_offset=2 => T-2 (3rd from end) => idx=-3
                    target_idx = -(target_offset + 1)
                    history = all_interactions[:target_idx]
                    target_item = all_interactions[target_idx]
                    
                    # Get instruction (if available)
                    instruction = None
                    if hasattr(self, 'instructions') and self.instructions:
                        instruction = self.instructions.get(user_id, None)
                    
                    # Construct candidate set (target + negatives)
                    candidates = self._construct_candidates(
                        user_id=user_id,
                        target_item=target_item,
                        history=history
                    )
                    
                    # Debug: Training round details
                    if self.debug:
                        self._log_debug(f"\n{'='*80}")
                        self._log_debug(f"🔥 TRAINING Round {warmup_round+1}/{self.warmup_rounds} - User {user_id}")
                        self._log_debug(f"{'='*80}")
                        self._log_debug(f"Target: T-{target_offset} (item {target_item})")
                        self._log_debug(f"History length: {len(history)}")
                        self._log_debug(f"Candidates: {len(candidates)} items")
                    
                    # Three-stage recommendation
                    try:
                        ranked_items, details = self.agent.rerank(
                            user_id=user_id,
                            candidates=candidates,
                            instruction=instruction,
                            debug_logger=self._log_debug if self.debug else None  # Pass debug_logger
                        )
                        
                        # Check target position
                        if target_item not in ranked_items:
                            if self.debug:
                                msg = f"  [Training] User {user_id}: Target {target_item} not in ranked_items!"
                                print(msg)
                                self._log_debug(msg)
                            continue
                        
                        target_pos = ranked_items.index(target_item)
                        
                        if self.debug:
                            msg = f"  [Training] User {user_id}: Target {target_item} ranked at position {target_pos+1}/{len(ranked_items)}"
                            print(msg)
                            self._log_debug(msg)
                            self._log_debug(f"\nRanking: {ranked_items[:5]}... (showing top-5)")
                        
                        # Generate feedback and update memory
                        if target_pos < 10:  # top-10 triggers Stage-W (relaxed condition to increase training success rate)
                            feedback = {
                                'action': 'CLICK',
                                'item_id': target_item,
                                'position': target_pos
                            }
                            
                            if self.debug:
                                self._log_debug(f"\n🔄 Stage-W: Updating memories...")
                            
                            # Stage-W: Write memory (extract correct parameters)
                            write_result = self.agent.write(
                                user_id=user_id,
                                feedback=feedback,
                                recent_facets=details.get('facets', []),
                                pruned_subgraph=details.get('pruned_subgraph', None),
                                debug_logger=self._log_debug if self.debug else None  # Pass debug_logger
                            )
                            
                            n_patches = write_result.get('stats', {}).get('total_applied', 0)
                            stats = write_result.get('stats', {})
                            
                            if self.debug:
                                msg = f"  [Training] User {user_id}: Stage-W applied {n_patches} patches"
                                print(msg)
                                self._log_debug(msg)
                                self._log_debug(f"  User updated: {stats.get('user_applied', 0)}")
                                self._log_debug(f"  Item updated: {stats.get('item_applied', 0)}")
                                self._log_debug(f"  Neighbors updated: {stats.get('neighbor_applied', 0)}")
                            
                            n_updates += 1
                        else:
                            if self.debug:
                                print(f"  [Training] User {user_id}: Target not in top-5, skipping Stage-W")
                    
                    except Exception as e:
                        if self.use_pregenerated_candidates:
                            raise
                        if self.debug:
                            print(f"  [Training] Error for user {user_id}: {e}")
                            import traceback
                            traceback.print_exc()
                        continue
            
            print(f"  ✓ {n_updates}/{len(eval_user_ids)} users updated memories")
        
        print(f"\n{'='*80}")
        print(f"🔥 MemRec Training Complete!")
        print(f"{'='*80}\n")
    
    def _generate_feedback(
        self,
        user_id: int,
        target_item: int,
        ranked_items: list[int],
        mode: str
    ) -> Dict:
        """
        Generate feedback information
        
        Args:
            user_id: User ID
            target_item: Target item
            ranked_items: Sorted item list
            mode: 'gt' or 'random'
            
        Returns:
            Feedback dictionary
        """
        if mode == 'gt':
            # Ground-truth: if target item is in top-5, CLICK; otherwise skip
            if target_item in ranked_items[:5]:
                return {
                    'action': 'CLICK',
                    'item_id': target_item,
                    'position': ranked_items.index(target_item)
                }
            else:
                # Optional: record as SKIP
                return None
        
        elif mode == 'random':
            # Randomly select a candidate and randomly generate feedback
            selected_item = random.choice(ranked_items[:5])  # Only select from top-5
            action = random.choice(['CLICK', 'SKIP'])
            return {
                'action': action,
                'item_id': selected_item,
                'position': ranked_items.index(selected_item)
            }
        
        return None
    
    def _compute_metrics(self, ranking_positions: list, n_users: int) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            ranking_positions: List of ranking positions (0-indexed)
            n_users: Number of users
            
        Returns:
            Metrics dictionary
        """
        metrics = {}
        
        if len(ranking_positions) == 0:
            # No valid results
            for k in self.topk:
                for metric in self.metrics:
                    metrics[f'{metric}@{k}'] = 0.0
            # metrics['MRR'] = 0.0  # MRR disabled for consistency
            return metrics
        
        positions = np.array(ranking_positions)
        
        # Hit@K and NDCG@K
        for k in self.topk:
            if 'Hit' in self.metrics:
                hit_at_k = np.mean(positions < k)
                metrics[f'Hit@{k}'] = float(hit_at_k)
            
            if 'NDCG' in self.metrics:
                # NDCG: 1/log2(pos+2) if pos < k, else 0
                ndcg_scores = np.where(
                    positions < k,
                    1.0 / np.log2(positions + 2),
                    0.0
                )
                metrics[f'NDCG@{k}'] = float(np.mean(ndcg_scores))
        
        # Note: MRR metric is disabled for consistency with baseline models
        # # MRR (always compute)
        # mrr = np.mean(1.0 / (positions + 1))
        # metrics['MRR'] = float(mrr)
        
        return metrics
    
    def train(self, save_dir: str = None) -> Dict:
        """
        Train (MemRec doesn't need training, directly return empty results)
        
        Returns:
            Empty dictionary (MemRec has no training phase)
        """
        # MemRec doesn't need traditional training (already builds memory through training phase)
        
        # Save save_dir for use by test()
        self.save_dir = save_dir
        
        # Return empty results, actual evaluation happens in test()
        return {
            'valid': {},
            'metrics': {}
        }
    
    def test(self, parallel: bool = False, n_workers: int = 16) -> Dict:
        """Test (standard interface)"""
        print("\n" + "="*80)
        print("Testing MemRec on test set")
        print("="*80)
        
        # Evaluation
        save_dir = getattr(self, 'save_dir', None)
        test_metrics = self.evaluate(split='test', save_dir=save_dir, parallel=parallel, n_workers=n_workers)
        
        # Save memory storage (optional)
        if save_dir:
            memory_path = Path(save_dir) / "memory.jsonl"
            self.agent.storage.save_to_jsonl(str(memory_path), dataset=self.dataset)
            
            # Output save information
            stats = self.agent.get_stats()
            print(f"Memory saved to {memory_path} ({stats['storage']['n_updates']} entries: {stats['storage']['n_users']} users, {stats['storage']['n_items']} items)")
            
            # If LLM conversations saving is enabled, output save information
            if self.save_conversations and self.conversation_file:
                conversation_path = Path(self.conversation_file)
                if conversation_path.exists():
                    # Count conversations
                    try:
                        with open(conversation_path, 'r') as f:
                            n_conversations = sum(1 for _ in f)
                        print(f"LLM conversations saved to {conversation_path} ({n_conversations} conversations)")
                    except:
                        print(f"LLM conversations saved to {conversation_path}")
        
        return test_metrics
