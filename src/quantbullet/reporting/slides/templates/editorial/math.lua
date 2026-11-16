-- Adds math-font.html (the theme's math font, written by SlideTemplate.install) to the page
-- header when the deck contains a formula and draws formulas as MathML (qb-math: mathml).

local mathml = false
local found = false

return {
  {
    Meta = function(meta)
      mathml = meta["qb-math"] ~= nil and pandoc.utils.stringify(meta["qb-math"]) == "mathml"
    end,
  },
  {
    Math = function(el)
      found = true
    end,
    Pandoc = function(doc)
      if mathml and found then
        quarto.doc.include_file("in-header", "math-font.html")
      end
    end,
  },
}
