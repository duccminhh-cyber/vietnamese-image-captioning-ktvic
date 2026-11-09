# Biên dịch proposal

Template ACL dùng chế độ `final`, A4 hai cột, XeLaTeX và BibTeX.
Không thu nhỏ lề hoặc cỡ chữ để nhồi nội dung. Font TeX Gyre Termes (Times tương
thích có tiếng Việt), fallback Times New Roman.

Tại thư mục này:

```powershell
xelatex -interaction=nonstopmode -halt-on-error proposal.tex
bibtex proposal
xelatex -interaction=nonstopmode -halt-on-error proposal.tex
xelatex -interaction=nonstopmode -halt-on-error proposal.tex
```

Kết quả `proposal.pdf`; bản bàn giao đặt tên `Nhom10_DeTai05_Proposal.pdf`.
Với Overleaf, upload mọi tệp trong thư mục này, chọn `proposal.tex` làm main
document và XeLaTeX làm compiler. `references.bib` quản lý trích dẫn tác giả–năm.

Nguồn template: https://github.com/acl-org/acl-style-files .
`acl.sty`, `acl_natbib.bst` được giữ nguyên từ nguồn template;
`natbib.sty` là phụ thuộc đi kèm bản build local, giữ nguyên thông báo bản quyền.
