// MathJax configuration for the `pymdownx.arithmatex` extension in `generic: true` mode,
// which wraps formulas in the script tags matched below rather than leaving raw TeX.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};
