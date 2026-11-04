# Hướng dẫn triển khai vào TranNgocMinh.github.io

## Quyền GitHub

Kết nối GitHub trong ChatGPT hiện chỉ đọc được repository và bị GitHub từ chối quyền ghi (HTTP 403). Vì vậy cần thực hiện thao tác đẩy mã nguồn bằng chính tài khoản GitHub của bạn.

## Cách an toàn — dùng Git

1. Tải bộ ZIP portfolio và giải nén.
2. Mở PowerShell ở một thư mục làm việc.
3. Clone repository:

```powershell
git clone https://github.com/TranNgocMinh/TranNgocMinh.github.io.git
cd TranNgocMinh.github.io
```

4. **Giữ nguyên hai file `myconvnext.py`, `myconvnext_train.py` hiện có**. Sao chép các file/folder trong ZIP (`index.html`, `assets`, `demos`, `favicon.svg`, `robots.txt`, `sitemap.xml`, `.nojekyll`, `README.md`) vào repository đã clone. File README cũ sẽ được thay bằng hướng dẫn website; nếu muốn giữ nguyên, hãy lưu nó trước dưới tên `README_CONVNEXT_LEGACY.md`.
5. Xem lại thay đổi:

```powershell
git status
git diff --stat
git add index.html assets demos favicon.svg robots.txt sitemap.xml .nojekyll README.md
git commit -m "Build freelance web developer portfolio"
git push origin main
```

6. Vào **Settings → Pages → Build and deployment**. Chọn `Deploy from a branch`, `main`, `/(root)` và Save.
7. Mở `https://tranngocminh.github.io/`, thử cả trên máy tính và điện thoại.

## Kiểm thử thủ công

- Menu di động đóng/mở và điều hướng đến các phần.
- Bộ lọc All / Live demos / Source projects chạy đúng.
- Taskboard thêm, sửa, xóa, đổi trạng thái, tìm kiếm và lưu khi refresh trang.
- Quote Builder cập nhật số giờ và ngân sách khi chọn các tùy chọn.
- Tất cả liên kết dẫn đúng nơi.
- Kiểm tra trên Chrome và điện thoại trước khi đưa liên kết lên Upwork.

## Sau khi có Upwork

Đổi dòng `Upwork freelancer profile: coming soon` trong `index.html` thành liên kết hồ sơ Upwork đã xác thực. Trước hợp đồng, nên tiếp tục trao đổi với khách hàng trong Upwork.
