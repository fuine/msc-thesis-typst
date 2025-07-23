#{
  import "@preview/wut-thesis:0.1.1": (
    acknowledgements, appendix, figure-outline, table-outline, wut-thesis,
  )
  import "utils.typ": flex-caption-styles, glossary-outline
  import "glossary.typ": glossary
  import "@preview/glossarium:0.5.6": make-glossary, register-glossary
  import "@preview/drafting:0.2.2": note-outline, set-margin-note-defaults

  show: make-glossary
  register-glossary(glossary)
  show: flex-caption-styles

  /** Drafting

    Set the boolean variables `draft` and `in-print` inside utils.typ.

    The "draft" variable is used to show DRAFT in the header and the title and TODOs.
    This should be true until the final version is handed-in.

    The "in-print" variable is used to generate a PDF file for physical printing (it adds
    bigger margins on the binding part of the page). If you want to create a PDF file for
    reading on screen (e.g. to upload to onedrive for the final thesis hand-in) set this
    variable to false.

  **/
  let draft = false
  let in-print = false
  set-margin-note-defaults(hidden: not draft)

  // Set the languages of your studies and thesis
  let lang = (
    // language in which you studied, influences the language of the titlepage
    studies: "pl",
    // language in which your thesis is written, influences the rest of the text (i.e.
    // abstracts order, captions/references supplements, hyphenation etc)
    thesis: "en",
  )

  show: wut-thesis.with(
    draft: draft,
    in-print: in-print,
    lang: lang,
    // Adjust the following fields accordingly to your thesis
    titlepage-info: (
      thesis-type: "master", // or "master"
      program: "Informatyka", // or ""
      specialisation: "Inżynieria Systemów Informatycznych",
      institute: "Instytut Informatyki",
      supervisor: "dr hab. inż. Robert M. Nowak",
      advisor: none, // or `none` if there were no advisors
      faculty: "weiti", // or "meil"
      index-number: "261479",
      date: datetime(year: 2018, month: 5, day: 31),
    ),
    author: "Łukasz Neumann",
    // Note that irregardless of the language of your thesis you need to fill in all the
    // fields below - both *-en and *-pl
    title: (
      en: [Heuristic hyperparameter optimization\ for neural networks],
      pl: [Heurystyczne strojenie hiperparametrów\ sieci neuronowych],
    ),
    abstract: (
      en: [
        // max 1 page
        Hyperparameter optimization plays an important role in creation of the robust
        neural network model. Correctly performed tuning should result in better
        classification quality, faster convergence of the network's optimizer, as well
        as smaller tendencies to overfit.

        This thesis explores the usage of selected heuristic algorithms: Covariance
        Matrix Adaptation Evolution Strategy (CMA-ES), Differential Evolution Strategy
        (DES) and jSO for the hyperparameter optimization problem. Hyperparameters for
        two architectures are tuned: Multilayer Perceptron (MLP) and Convolutional
        Neural Network (CNN). Three real-life datasets are used as a basis for the
        classification task.

        Variety of evaluation methods for the hyperparameter optimization process is
        presented. These methods allow for easier characterization of the tuning process
        itself, as well as neural networks created using tuned hyperparameters.
        Moreover, a technique for comparison between different implementations of the
        same heuristic algorithm is described.

        Results indicate, that all three heuristic algorithms can be successfully used as
        a method for hyperparameter optimization. Notably, similar performance is
        observed amongst all optimizers. Classifiers created using the best individuals
        found during the training process can outperform a reference classifier, when
        such comparison is being made.
      ],
      pl: [
        // max 1 page
        Optymalizacja wartości hiperparametrów jest jedną z kluczowych części procesu
        tworzenia wartościowej sieci neuronowej. Poprawnie przeprowadzone strojenie
        powinno skutkować lepszą jakością klasyfikacji, szybszym zbieganiem metody
        uczenia sieci do optimum, a także mniejszą podatnością modelu na zbytnie
        dopasowanie do danych trenujących.

        Niniejsza praca opisuje wykorzystanie trzech algorytmów heurystycznych:
        Covariance Matrix Adaptation Evolution Strategy (CMA-ES), Differential Evolution
        Strategy (DES) i jSO na potrzeby problemu strojenia hiperparametrów. Badania
        przeprowadzane zostały dla dwóch architektur: perceptronu wielowarstwowego
        i splotowej sieci neuronowej. Ocena modeli odbyła się na podstawie klasyfikacji
        trzech zbiorów danych.

        Praca zawiera opis metod ewaluacji procesu strojenia hiperparametrów.
        Pozwalają one na łatwiejszy opis samej optymalizacji, a także sieci neuronowych
        stworzonych na podstawie znalezionych wartości hiperparametrów. Dodatkowo
        przedstawiona jest technika porównania różnych implementacji tego samego
        algorytmu heurystycznego.

        Wyniki pokazują, że wszystkie zbadane algorytmy heurystyczne mogą być z
        powodzeniem użyte do strojenia hiperparametrów sieci neuronowych. W
        szczególności, wszystkie optymalizatory cechują się podobną jakością strojenia.
        Klasyfikatory stworzone na podstawie najlepszych znalezionych zestawów
        hiperparametrów osiągają lepsze wyniki od klasyfikatora odniesienia, w
        przypadkach, gdzie takie porównanie miało miejsce.
      ],
    ),
    keywords: (
      en: (
        "evolutionary algorithm",
        "optimization",
        "tuning",
        "hyperparameter",
        "neural network",
      ),
      pl: (
        "algorytm ewolucyjny",
        "optymalizacja",
        "strojenie",
        "hiperparametr",
        "sieć neuronowa",
      ),
    ),
  )

  // --- Custom Settings ---
  // if you want to override any settings from the template here is the place to do so,
  // e.g.:
  // set text(font: "Comic Sans MS")
  // disable numbering and outlining for the level 4 and 5 headings
  let heading-selectors = (4, 5).map(x => heading.where(level: x))
  show selector.or(..heading-selectors): set heading(outlined: false, numbering: none)
  // booktabs-like table decorations
  set table(
    stroke: (x, y) => (
      top: if y <= 1 { 1pt } else { 0pt },
      bottom: 1pt,
    ),
    inset: (x: 5pt, y: 3pt),
  )

  // --- Main Chapters ---
  include "content/ch1_intro.typ"
  include "content/ch2_methods.typ"
  include "content/ch3_results.typ"
  include "content/ch4_summary.typ"

  // --- Acknowledgements ---
  // comment out if not needed
  // acknowledgements[
  //   We gratefully acknowledge Poland's high-performance Infrastructure PLGrid
  //   #text(fill: red)[(wybierz właściwy ośrodek z listy: ACK Cyfronet AGH, PCSS, CI TASK,
  //     WCSS)] for providing computer facilities and support within computational grant no
  //   #text(fill: red)[(numer grantu)]
  //   #todo[Numer grantu i typ ośrodka]
  // ]

  // --- Bibliography ---
  bibliography("bibliography.bib", style: "ieee")

  // List of Acronyms - comment out, if not needed (no abbreviations were used).
  // glossary-outline(glossary)

  // List of figures - comment out, if not needed.
  figure-outline()

  // List of tables - comment out, if not needed.
  table-outline()

  // --- Appendices ---
  appendix(lang.thesis, include "content/ch5_appendix.typ")

  if draft {
    set heading(numbering: none)
    note-outline(title: "TODOs")
  }
}
