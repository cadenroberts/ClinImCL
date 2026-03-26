.PHONY: all clean

EXPECTED_FIGS = figures/epoch1_projections.png figures/epoch20_projections.png \
	figures/epoch40_projections.png figures/linearprobe_cm.png \
	figures/linearprobe_roc.png figures/oasisbrains.png \
	figures/test_confusion.png figures/test_projections.png

all:
	@missing=0; for f in $(EXPECTED_FIGS); do \
		if [ ! -f "$$f" ]; then echo "[missing] $$f"; missing=1; fi; \
	done; \
	if [ "$$missing" = "0" ]; then echo "All expected figures present."; fi

clean:
	rm -rf __pycache__ visualizations/
	rm -f *.pyc .cookies-*.txt oasis*.csv subset_* _tmp.csv
