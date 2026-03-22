.PHONY: pdf all clean

pdf:
	cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex
	cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex

all: pdf

clean:
	rm -f report/report.aux report/report.log report/report.out
