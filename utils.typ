#import "@preview/drafting:0.2.2": inline-note
#import "@preview/lovelace:0.3.0": *
#import "@preview/subpar:0.2.2"

/// Create the outline of glossary terms with localized title
#let glossary-outline(glossary) = {
  context {
    let lang = text.lang
    let glossary-text = if lang == "en" { "List of Symbols and Abbreviations" } else {
      "Wykaz symboli i skrótów"
    }
    heading(numbering: none, glossary-text)
    show figure: it => [#v(-1em) #it #v(-1em)]
    print-glossary(glossary, show-all: true, disable-back-references: true)
  }
}

/// Create a TODO bubble with the note being outlined in the TODO outline
#let todo(it) = [
  #let caution-rect = rect.with(inset: 1em, radius: 0.5em)
  #inline-note(rect: caution-rect, stroke: color.fuchsia, fill: color.fuchsia.lighten(
    80%,
  ))[
    #align(center + horizon)[#text(fill: color.fuchsia, weight: "extrabold")[TODO:] #it]
  ]
]

/// State for the `flex-caption` function
#let in-outline = state("in-outline", false)

/// Show rule for the `flex-caption` function to work properly
#let flex-caption-styles = rest => {
  show outline: it => {
    in-outline.update(true)
    it
    in-outline.update(false)
  }
  rest
}

/// When used, provides different captions for in-document caption (`long`) and
//  in-outline (`short`) caption
#let flex-caption(long, short) = (
  context (
    if in-outline.get() {
      short
    } else {
      long
    }
  )
)

/// Table with footnotes attached to the bottom of it
#let table-with-notes(notes: none, ..args) = layout(size => {
  let tbl = table(..args)
  let w = measure(..size, tbl).width
  stack(dir: ttb, spacing: 0.5em, tbl, align(left + top, block(width: w, notes)))
})

/// `Algorithm` environment based on the `lovelace` package
#let algorithm(content, caption: none, ..args) = {
  figure(
    pseudocode-list(..args, booktabs: true, hooks: .5em, content),
    caption: caption,
    kind: "algorithm",
    supplement: [Algorithm],
  )
}

/// Comment in the `Algorithm` environment
#let comment(body) = {
  text(size: .85em, fill: gray.darken(30%), sym.triangle.stroked.r + sym.space + body)
}
/// Shorthand for latex'x \mathbf
#let ub(content) = math.upright(math.bold(content))

/// Breakable subfigures based on the `subpar` package
#let multipage-subfigures(caption: none, label: none, ..figures) = {
  show figure: set block(breakable: true)
  subpar.grid(caption: caption, columns: 1, label: label, ..figures
    .pos()
    .map(fig => figure(image("images/" + fig.first()), caption: fig.last())))
}
