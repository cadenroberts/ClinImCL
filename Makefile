.PHONY: pdf all clean

pdf:
	pdflatex -interaction=nonstopmode -halt-on-error report.tex
	pdflatex -interaction=nonstopmode -halt-on-error report.tex

all: pdf

clean:
	rm -f report.aux report.log report.out
