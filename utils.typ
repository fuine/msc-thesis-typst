#import "@preview/drafting:0.2.2": inline-note
#import "@preview/glossarium:0.5.6": make-glossary, register-glossary, print-glossary, gls, glspl
#import "@preview/lovelace:0.3.0": *
#import "@preview/theorion:0.3.3": *
#import "@preview/subpar:0.2.2"

#import cosmos.rainbow: *
#let definition = definition.with(fill: blue.darken(10%))
#let definition-box = definition-box.with(fill: blue.darken(10%))
#let (property-counter, property-box, property, show-property) = make-frame(
  "property",
  "Property",  // supplement, string or dictionary like `(en: "Theorem")`, or `theorion-i18n-map.at("theorem")` for built-in i18n support
  inherited-levels: 2,  // useful when you need a new counter
  inherited-from: heading,  // heading or just another counter
  render: render-fn.with(fill: orange)
)



#let glossary-outline(glossary) = {
  context {
    let lang = text.lang
    let glossary-text = if lang == "en" { "List of Symbols and Abbreviations" } else { "Wykaz symboli i skrótów" }
    heading(numbering: none, glossary-text)
    show figure: it => [#v(-1em) #it #v(-1em)]
    print-glossary(glossary, show-all: true, disable-back-references: true)
  }
}

#let figure-outline() = {
  context {
    let lang = text.lang
    let figures-text = if lang == "en" { "List of Figures" } else { "Spis rysunków" }
    outline(
      title: figures-text,
      target: figure.where(kind: image),
    )
  }
}

#let table-outline() = {
  context {
    let lang = text.lang
    let tables-text = if lang == "en" { "List of Tables" } else { "Spis tabel" }
    outline(
      title: tables-text,
      target: figure.where(kind: table),
    )
  }
}

#let todo(it) = [
  #let caution-rect = rect.with(inset: 1em, radius: 0.5em)
  #inline-note(rect: caution-rect, stroke: color.fuchsia, fill: color.fuchsia.lighten(80%))[
    #align(center + horizon)[#text(fill: color.fuchsia, weight: "extrabold")[TODO:] #it]
  ]
]

#let silentheading(level, body) = [
  #heading(outlined: false, level: level, numbering: none, bookmarked: true)[#body]
]

#let in-outline = state("in-outline", false)

#let flex-caption-styles = rest => {
  show outline: it => {
    in-outline.update(true)
    it
    in-outline.update(false)
  }
  rest
}

#let flex-caption(long, short) = (
  context (
    if in-outline.get() {
      short
    } else {
      long
    }
  )
)

#let table-with-notes(notes: none, ..args) = layout(size => {
  let tbl = table(..args)
  let w = measure(..size, tbl).width
  stack(dir: ttb, spacing: 0.5em, tbl, align(left + top, block(width: w, notes)))
})

#let code-listing-figure(caption: none, content) = {
  figure(
    caption: caption,
    rect(
      stroke: (y: 1pt + black),
      align(left, content),
    ),
  )
}

#let code-listing-file(filename, caption: none) = {
  let extension = filename.split(".").last()
  code-listing-figure(raw(block: true, lang: extension, read("code_snippets/" + filename)), caption: caption)
}

#let algorithm(content, caption: none, ..args) = {
  figure(
    pseudocode-list(..args, booktabs: true, hooks: .5em, content),
    caption: caption,
    kind: "algorithm",
    supplement: [Algorithm],
  )
}
#let comment(body) = {
  text(size: .85em, fill: gray.darken(30%), sym.triangle.stroked.r + sym.space + body)
}
#let larrow = sym.arrow.l
#let ub(content) = math.upright(math.bold(content))

#let multipage-subfigures(caption: none, label: none, ..figures) = {
  show figure: set block(breakable: true)
  subpar.grid(
    caption: caption,
    columns: 1,
    label: label,
    ..figures.pos().map(fig => figure(image("images/" + fig.first()), caption: fig.last()))
  )
}
