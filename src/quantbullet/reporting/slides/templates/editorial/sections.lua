-- Level-1 headings become section divider slides.
--
-- Metadata it reads (all optional):
--   qb-section-background   background color of section slides (default #0b3d6e)
--   qb-appendix-background  background of sections with class .appendix-section (default #334155)
--   qb-number-sections      true: prefix "01", "02", ... as a small label above each title;
--                           a section with class .unnumbered is skipped
-- An explicit {background-color="..."} on a heading wins.

local section_bg = "#0b3d6e"
local appendix_bg = "#334155"
local number_sections = false
local count = 0

local function read_meta(meta)
  if meta["qb-section-background"] then
    section_bg = pandoc.utils.stringify(meta["qb-section-background"])
  end
  if meta["qb-appendix-background"] then
    appendix_bg = pandoc.utils.stringify(meta["qb-appendix-background"])
  end
  if meta["qb-number-sections"] == true then
    number_sections = true
  end
end

local function has_class(el, name)
  for _, c in ipairs(el.classes) do
    if c == name then return true end
  end
  return false
end

local function decorate(el)
  if el.level ~= 1 then return nil end
  if not has_class(el, "section-slide") then
    el.classes:insert("section-slide")
  end
  if not el.attributes["background-color"] then
    el.attributes["background-color"] = has_class(el, "appendix-section") and appendix_bg or section_bg
  end
  if number_sections and not has_class(el, "unnumbered") then
    count = count + 1
    local label = pandoc.Span({pandoc.Str(string.format("%02d", count))}, {class = "section-num"})
    el.content:insert(1, label)
  end
  return el
end

return {
  { Meta = read_meta },
  { Header = decorate },
}
