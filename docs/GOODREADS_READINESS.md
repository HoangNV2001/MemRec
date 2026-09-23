# Goodreads dataset readiness audit

**Audit date:** 2026-09-23

**Scope:** bounded schema/sample audit only; no full scan, no GPU, no model run

**Decision:** **deferred — not ready for temporal next-item evaluation**

## 1. Local snapshot

`data/goodreads/` hiện có khoảng 1,2 GiB:

- 23 metadata shards: `book1-100k.csv` đến `book4000k-5000k.csv`;
- 7 rating shards: `user_rating_0_to_1000.csv` đến
  `user_rating_6000_to_11000.csv`.

Audit đọc header và tối đa 2.000 record đầu của mỗi shard, tổng cộng 46.000
metadata record và 14.000 rating record. Đây chỉ là readiness check, không phải
thống kê toàn dataset.

## 2. Findings

### Metadata

- Các shard đều có `Id` và `Name`, nhưng column order không thống nhất.
- Schema có 18–20 cột; một số shard có `Description` và
  `Count of text reviews`, một số không.
- Tên page field thay đổi giữa `pagesNumber` và `PagesNumber`.
- 46.000 record mẫu parse được; không thấy blank `Id`/`Name` hay duplicate
  title trong từng prefix sample. Điều này **không chứng minh** title unique trên
  toàn bộ metadata snapshot hoặc giữa các shard.

### Ratings/interactions

Mọi rating shard có đúng schema:

```text
ID,Name,Rating
```

- `ID` có vẻ là user identifier.
- Item chỉ được biểu diễn bằng title `Name`; không có stable book ID để join
  trực tiếp với metadata `Id`.
- `Rating` là categorical text: `did not like it`, `it was ok`, `liked it`,
  `really liked it`, `it was amazing`, và ở sample có cả
  `This user doesn't have any rating`.
- Không có timestamp, review time, sequence index hay field chronology khác.
- Sample phát hiện một số duplicate `(user, title)` và user ID có thể xuất hiện
  ở biên hai shard liên tiếp, nên loader tương lai phải merge/de-duplicate xuyên
  shard thay vì giả định partition rời nhau.

### Compatibility với code hiện tại

`configs/memrec_instructrec-goodreads.yaml` và
`src/memory/domain_rules/goodreads_rules.py` thuộc pipeline
`instructrec-goodreads`; chúng không chứng minh raw snapshot này có temporal
contract hoặc tương thích trực tiếp. Đặc biệt, rule dùng `recency_days` nhưng raw
ratings hiện không có thời gian.

## 3. Vì sao chưa dùng cho thesis core

Temporal transition graph cần thứ tự interaction strict-past. Row order trong
CSV/shard không được phép coi là chronology nếu dataset không định nghĩa như
vậy. Nếu tự gán thứ tự theo row, graph `A → B` sẽ phản ánh serialization của
file thay vì hành vi đọc sách và tạo construct invalidity/leakage khó kiểm soát.

Join theo title cũng có rủi ro gộp nhiều edition hoặc các sách trùng tên. Vì
rating rows không có metadata `Id`, hiện chưa có cách leakage-safe và
identity-safe để dựng chuỗi item.

Do đó snapshot hiện tại không trả lời cùng RQ với Amazon Books và MovieLens.
Thêm nó vào bảng chính lúc này sẽ làm external-validity claim yếu hơn thay vì
mạnh hơn.

## 4. Promotion gate

Goodreads chỉ được đưa vào temporal scope nếu tìm được source/companion table
có tối thiểu:

```text
user_id
stable_book_id
interaction_timestamp
rating_or_event
```

và `stable_book_id` join một-một hoặc many-to-one có quy tắc rõ với metadata.
Sau đó phải audit:

1. provenance/license/checksum;
2. timestamp unit/range và ties;
3. duplicate `(user, item, timestamp)`;
4. user/item counts và sequence-length distribution;
5. metadata coverage bằng stable ID;
6. global temporal split và strict-past replay;
7. eligible next-item events và graph coverage.

Chỉ khi audit pass mới viết protocol. Sau unit test phải smoke 20–30 event,
chạy CPU-only graph headroom trước, và chỉ dùng LLM/GPU nếu gate preregistered
pass. Toàn bộ graph/fusion parameter phải frozen-transfer từ study hiện tại;
không tune riêng trên Goodreads.

## 5. Vai trò tạm thời

- Giữ nguyên raw files trong `data/goodreads/`; chưa convert hoặc tạo artifact
  lớn.
- Không dùng nó như bằng chứng temporal/sequential trong thesis hiện tại.
- Có thể nhắc ở dataset selection appendix là một candidate dataset bị loại vì
  thiếu timestamp và stable item key.
- Nếu chỉ nghiên cứu static recommendation sau này, snapshot vẫn có thể hữu
  ích, nhưng đó là scope khác và không được ưu tiên trước M7 MovieLens
  graph-hard end-to-end.
