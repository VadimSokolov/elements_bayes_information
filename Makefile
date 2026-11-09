# Build the book: pdflatex, makeindex, pdflatex, pdflatex. Aux files go to $(BUILD), the PDF to the top level.
# Agents building in parallel pass their own directory: make BUILD=/tmp/eob_work/build-<name>
BOOK  := Elements_of_Bayes_INTERIOR
BUILD ?= build
LATEX := max_print_line=10000 pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD)
FAILED = { echo "build failed, see $(BUILD)/$(BOOK).log:"; grep -A3 '^!' $(BUILD)/$(BOOK).log | head -20; exit 1; }

.PHONY: pdf check clean

pdf:
	@mkdir -p $(BUILD)
	@$(LATEX) $(BOOK).tex > /dev/null || $(FAILED)
	@cd $(BUILD) && makeindex -q $(BOOK).idx
	@$(LATEX) $(BOOK).tex > /dev/null || $(FAILED)
	@$(LATEX) $(BOOK).tex > /dev/null || $(FAILED)
	@if [ "$(BUILD)" = build ]; then cp build/$(BOOK).pdf $(BOOK).pdf; fi
	@python3 tools/check.py --build $(BUILD)

check:
	@python3 tools/check.py --build $(BUILD)

clean:
	rm -rf build
