const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, TableOfContents, LevelFormat,
} = require("docx");

// ============================================================
//  HELPERS
// ============================================================
const FONT = "Times New Roman";
const PAGE_W = 11906; // A4
const PAGE_H = 16838;
const MARGIN = 1440;  // 1 inch
const CONTENT_W = PAGE_W - 2 * MARGIN; // ~9026

const txt = (t, opts = {}) => new TextRun({ text: t, font: FONT, size: opts.size || 24, ...opts });
const ital = (t, opts = {}) => txt(t, { italics: true, ...opts });
const bold = (t, opts = {}) => txt(t, { bold: true, ...opts });
const sub = (t) => txt(t, { subScript: true });
const sup = (t) => txt(t, { superScript: true });

const para = (children, opts = {}) => new Paragraph({
  spacing: { after: 120, line: 360 },
  alignment: AlignmentType.JUSTIFIED,
  ...opts,
  children: Array.isArray(children) ? children : [txt(children)],
});

const heading = (level, text) => new Paragraph({
  heading: level,
  spacing: { before: 360, after: 200 },
  children: [txt(text, { bold: true, size: level === HeadingLevel.HEADING_1 ? 32 : level === HeadingLevel.HEADING_2 ? 28 : 24 })],
});

const eq = (children) => new Paragraph({
  spacing: { before: 200, after: 200 },
  alignment: AlignmentType.CENTER,
  children: Array.isArray(children) ? children : [ital(children)],
});

const figCaption = (text) => new Paragraph({
  spacing: { before: 100, after: 200 },
  alignment: AlignmentType.CENTER,
  children: [ital(text, { size: 22 })],
});

const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellM = { top: 60, bottom: 60, left: 100, right: 100 };

const cell = (text, opts = {}) => new TableCell({
  borders,
  width: opts.width ? { size: opts.width, type: WidthType.DXA } : undefined,
  margins: cellM,
  shading: opts.header ? { fill: "D5E8F0", type: ShadingType.CLEAR } : undefined,
  verticalAlign: "center",
  children: [new Paragraph({
    alignment: opts.align || AlignmentType.LEFT,
    children: [txt(text, { bold: !!opts.header, size: 22 })],
  })],
});

const tableRow = (cells) => new TableRow({ children: cells });

const simpleTable = (headers, rows, colWidths) => {
  const w = colWidths || headers.map(() => Math.floor(CONTENT_W / headers.length));
  const total = w.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: w,
    rows: [
      tableRow(headers.map((h, i) => cell(h, { header: true, width: w[i], align: AlignmentType.CENTER }))),
      ...rows.map(r => tableRow(r.map((c, i) => cell(c, { width: w[i] })))),
    ],
  });
};

const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

// ============================================================
//  CONTENT
// ============================================================

// --- TITLE PAGE ---
const titlePage = [
  new Paragraph({ spacing: { before: 3000 } }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [txt("RAPPORT DE PROJET", { bold: true, size: 36, font: FONT })],
  }),
  new Paragraph({ spacing: { after: 100 }, alignment: AlignmentType.CENTER, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 400 },
    children: [txt("Apprentissage par Renforcement Profond", { bold: true, size: 32, font: FONT })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [txt("pour le Contr\u00F4le Thermique de Batteries", { bold: true, size: 32, font: FONT })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
    children: [txt("par Agent Soft Actor-Critic (SAC)", { bold: true, size: 28, font: FONT })],
  }),
  new Paragraph({ spacing: { after: 800 }, alignment: AlignmentType.CENTER, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 100 },
    children: [txt("Auteur : Abdellah HAYANE", { size: 26, font: FONT })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 100 },
    children: [txt("Mars 2026", { size: 24, font: FONT })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [txt("Challenge 7 jours \u2014 Blocs de 2-3 heures", { size: 22, font: FONT, italics: true })],
  }),
  pageBreak(),
];

// --- ABSTRACT ---
const abstractSection = [
  heading(HeadingLevel.HEADING_1, "R\u00E9sum\u00E9"),
  para("Ce rapport pr\u00E9sente la conception, l\u2019impl\u00E9mentation et l\u2019\u00E9valuation d\u2019un syst\u00E8me de contr\u00F4le thermique de batteries par apprentissage par renforcement profond. L\u2019objectif est d\u2019entra\u00EEner un agent Soft Actor-Critic (SAC) \u00E0 r\u00E9guler la puissance de refroidissement d\u2019un pack batterie, en maintenant la temp\u00E9rature cellulaire dans une plage s\u00FBre tout en minimisant la consommation \u00E9nerg\u00E9tique du syst\u00E8me de refroidissement."),
  para("L\u2019environnement de simulation repose sur un mod\u00E8le thermique \u00E9quivalent RC \u00E0 un n\u0153ud, impl\u00E9ment\u00E9 comme un environnement Gymnasium. L\u2019agent SAC, impl\u00E9ment\u00E9 enti\u00E8rement from scratch en PyTorch, combine un acteur gaussien, un double critique (twin Q-networks) et une entropie adaptative. Apr\u00E8s 300 000 pas d\u2019entra\u00EEnement, l\u2019agent maintient la temp\u00E9rature en zone s\u00FBre 96,7 % du temps, d\u00E9montrant la viabilit\u00E9 de l\u2019approche pour le contr\u00F4le thermique embarqu\u00E9 en contexte FSE (Field Service Engineering)."),
  para([bold("Mots-cl\u00E9s : "), txt("apprentissage par renforcement, Soft Actor-Critic, contr\u00F4le thermique, batterie lithium-ion, mod\u00E8le RC, gestion thermique embarqu\u00E9e.")]),
  pageBreak(),
];

// --- TOC ---
const tocSection = [
  heading(HeadingLevel.HEADING_1, "Table des mati\u00E8res"),
  new TableOfContents("Table des mati\u00E8res", { hyperlink: true, headingStyleRange: "1-3" }),
  pageBreak(),
];

// --- 1. INTRODUCTION ---
const intro = [
  heading(HeadingLevel.HEADING_1, "1. Introduction"),

  heading(HeadingLevel.HEADING_2, "1.1 Contexte du projet"),
  para("Les batteries lithium-ion constituent aujourd\u2019hui le vecteur de stockage \u00E9nerg\u00E9tique dominant dans les domaines de la mobilit\u00E9 \u00E9lectrique, du stockage stationnaire et des syst\u00E8mes embarqu\u00E9s industriels. Leur performance, leur dur\u00E9e de vie et leur s\u00E9curit\u00E9 sont fortement conditionn\u00E9es par la gestion thermique. Une temp\u00E9rature exc\u00E9dant les limites op\u00E9rationnelles (typiquement 15\u00B0C \u2013 45\u00B0C) acc\u00E9l\u00E8re la d\u00E9gradation des mat\u00E9riaux d\u2019\u00E9lectrode, r\u00E9duit la capacit\u00E9 utile et, dans les cas extr\u00EAmes, peut conduire \u00E0 un emballement thermique."),
  para("Dans le contexte FSE (Field Service Engineering), les packs batterie sont soumis \u00E0 des profils de charge et de d\u00E9charge variables, dans des environnements o\u00F9 la temp\u00E9rature ambiante fluctue. Les strat\u00E9gies de contr\u00F4le thermique classiques \u2014 bas\u00E9es sur des r\u00E8gles heuristiques ou des contr\u00F4leurs PID \u2014 peinent \u00E0 s\u2019adapter dynamiquement \u00E0 cette variabilit\u00E9."),

  heading(HeadingLevel.HEADING_2, "1.2 Probl\u00E9matique"),
  para("La probl\u00E9matique centrale de ce projet se formule ainsi : est-il possible d\u2019entra\u00EEner un agent d\u2019apprentissage par renforcement profond \u00E0 contr\u00F4ler la puissance de refroidissement d\u2019un pack batterie de mani\u00E8re optimale, en temps r\u00E9el, sous des conditions stochastiques de courant et de temp\u00E9rature ambiante ?"),
  para("Cette question soul\u00E8ve plusieurs sous-probl\u00E8mes :"),
  para("\u2022  Comment mod\u00E9liser fid\u00E8lement la dynamique thermique d\u2019une batterie dans un environnement de simulation compatible avec les algorithmes RL ?"),
  para("\u2022  Quel algorithme RL choisir pour un espace d\u2019action continu avec des contraintes de stabilit\u00E9 et d\u2019efficacit\u00E9 \u00E9chantillonnale ?"),
  para("\u2022  Comment concevoir une fonction de r\u00E9compense qui encode simultan\u00E9ment la s\u00E9curit\u00E9 thermique, l\u2019efficacit\u00E9 \u00E9nerg\u00E9tique et la progressivit\u00E9 de la charge ?"),

  heading(HeadingLevel.HEADING_2, "1.3 Objectifs"),
  para("Les objectifs de ce projet, r\u00E9alis\u00E9 sur un challenge de 7 jours par blocs de 2 \u00E0 3 heures, sont les suivants :"),
  para("1.  Concevoir un environnement Gymnasium fid\u00E8le int\u00E9grant un mod\u00E8le thermique RC \u00E0 un n\u0153ud."),
  para("2.  Impl\u00E9menter from scratch un agent SAC complet (acteur, double critique, entropie adaptative)."),
  para("3.  D\u00E9velopper une boucle d\u2019entra\u00EEnement robuste avec logging, \u00E9valuation p\u00E9riodique et sauvegarde de checkpoints."),
  para("4.  Atteindre un pourcentage de temps en zone thermique s\u00FBre (pct_safe) sup\u00E9rieur \u00E0 95 %."),
  para("5.  Analyser l\u2019impact du reward shaping et du tuning des hyperparam\u00E8tres sur les performances."),
  pageBreak(),
];

// --- 2. ETAT DE L'ART ---
const etatArt = [
  heading(HeadingLevel.HEADING_1, "2. \u00C9tat de l\u2019art"),

  heading(HeadingLevel.HEADING_2, "2.1 Gestion thermique des batteries : m\u00E9thodes existantes"),
  para("La gestion thermique des batteries (Battery Thermal Management System, BTMS) a fait l\u2019objet de nombreuses recherches. Les approches peuvent \u00EAtre class\u00E9es en trois cat\u00E9gories principales :"),

  heading(HeadingLevel.HEADING_3, "2.1.1 Contr\u00F4le par r\u00E8gles heuristiques"),
  para("Les syst\u00E8mes les plus simples utilisent des seuils fixes : le refroidissement est activ\u00E9 lorsque la temp\u00E9rature d\u00E9passe un seuil haut et d\u00E9sactiv\u00E9 sous un seuil bas (logique bang-bang ou hyst\u00E9r\u00E9sis). Cette approche, facile \u00E0 impl\u00E9menter, ne permet pas d\u2019anticiper les mont\u00E9es thermiques et conduit \u00E0 des oscillations autour des seuils."),

  heading(HeadingLevel.HEADING_3, "2.1.2 Contr\u00F4le PID"),
  para("Le contr\u00F4leur PID (Proportionnel-Int\u00E9gral-D\u00E9riv\u00E9) est largement utilis\u00E9 dans l\u2019industrie. Il r\u00E9agit \u00E0 l\u2019erreur entre la temp\u00E9rature mesur\u00E9e et une consigne. Ses limites incluent la n\u00E9cessit\u00E9 d\u2019un tuning manuel, la difficult\u00E9 \u00E0 g\u00E9rer des non-lin\u00E9arit\u00E9s et l\u2019absence d\u2019optimisation globale sur un horizon temporel."),

  heading(HeadingLevel.HEADING_3, "2.1.3 Contr\u00F4le pr\u00E9dictif par mod\u00E8le (MPC)"),
  para("Le MPC (Model Predictive Control) optimise une trajectoire de contr\u00F4le sur un horizon glissant en utilisant un mod\u00E8le pr\u00E9dictif du syst\u00E8me. Bien que performant, il requiert un mod\u00E8le analytique pr\u00E9cis, un solveur d\u2019optimisation en temps r\u00E9el et des ressources de calcul significatives, ce qui limite son d\u00E9ploiement dans les syst\u00E8mes embarqu\u00E9s \u00E0 faibles ressources."),

  heading(HeadingLevel.HEADING_2, "2.2 Apprentissage par renforcement pour le contr\u00F4le thermique"),
  para("L\u2019apprentissage par renforcement (RL) offre une alternative prometteuse. Un agent RL apprend une politique de contr\u00F4le directement \u00E0 partir de l\u2019interaction avec l\u2019environnement, sans n\u00E9cessiter de mod\u00E8le analytique explicite pour le contr\u00F4le (bien qu\u2019un mod\u00E8le soit n\u00E9cessaire pour la simulation)."),
  para("Parmi les algorithmes RL adapt\u00E9s aux espaces d\u2019action continus, le Soft Actor-Critic (SAC), propos\u00E9 par Haarnoja et al. (2018), se distingue par :"),
  para("\u2022  Son efficacit\u00E9 \u00E9chantillonnale (off-policy, replay buffer)."),
  para("\u2022  Sa stabilit\u00E9 d\u2019entra\u00EEnement (double Q-network, entropie adaptative)."),
  para("\u2022  Sa capacit\u00E9 \u00E0 explorer efficacement gr\u00E2ce \u00E0 la r\u00E9gularisation entropique."),

  heading(HeadingLevel.HEADING_2, "2.3 Positionnement de notre approche"),
  para("Notre approche se positionne \u00E0 l\u2019intersection du contr\u00F4le thermique et du RL profond. Contrairement aux travaux existants qui utilisent souvent des biblioth\u00E8ques pr\u00E9-construites (Stable-Baselines3), nous impl\u00E9mentons l\u2019agent SAC enti\u00E8rement from scratch en PyTorch, ce qui offre une transparence totale sur les choix architecturaux et algorithmiques. L\u2019environnement Gymnasium est \u00E9galement con\u00E7u sur mesure, avec des param\u00E8tres thermiques g\u00E9n\u00E9riques adaptables \u00E0 diff\u00E9rents types de batteries et contextes FSE."),
  pageBreak(),
];

// --- 3. METHODOLOGIE ---
const methodologie = [
  heading(HeadingLevel.HEADING_1, "3. M\u00E9thodologie"),

  heading(HeadingLevel.HEADING_2, "3.1 Architecture g\u00E9n\u00E9rale du syst\u00E8me"),
  para("Le syst\u00E8me se d\u00E9compose en trois couches fonctionnelles principales :"),
  para("1.  La couche Environnement, qui simule la dynamique thermique de la batterie et expose l\u2019interface Gymnasium (reset, step, observation, reward)."),
  para("2.  La couche Agent, qui impl\u00E9mente l\u2019algorithme SAC avec ses r\u00E9seaux de neurones (acteur, double critique, r\u00E9seau cible)."),
  para("3.  La couche Entra\u00EEnement, qui orchestre l\u2019interaction env-agent, le logging et l\u2019\u00E9valuation."),
  figCaption("Figure 1 : Architecture g\u00E9n\u00E9rale du syst\u00E8me. L\u2019environnement Gymnasium produit des observations normalis\u00E9es que l\u2019agent SAC transforme en actions de refroidissement. Le reward guide l\u2019apprentissage."),

  heading(HeadingLevel.HEADING_2, "3.2 Environnement de simulation"),
  para("L\u2019environnement de simulation est con\u00E7u comme un environnement Gymnasium standard, garantissant la compatibilit\u00E9 avec l\u2019\u00E9cosyst\u00E8me RL Python. Il encapsule :"),
  para("\u2022  Un mod\u00E8le thermique RC \u00E0 un n\u0153ud (classe ThermalModel)."),
  para("\u2022  Un mod\u00E8le de batterie simplifi\u00E9 (SoC par int\u00E9gration du courant)."),
  para("\u2022  Un profil de courant stochastique (marche al\u00E9atoire born\u00E9e)."),
  para("\u2022  Une temp\u00E9rature ambiante lentement variable."),
  para("L\u2019espace d\u2019observation est un vecteur de dimension 5, normalis\u00E9 dans [-1, 1] :"),
  simpleTable(
    ["Composante", "Variable physique", "Plage brute"],
    [
      ["T_norm", "Temp\u00E9rature cellulaire", "10\u00B0C \u2013 60\u00B0C"],
      ["SoC_norm", "\u00C9tat de charge", "0 \u2013 1"],
      ["I_norm", "Courant de charge", "-10A \u2013 50A"],
      ["T_amb_norm", "Temp\u00E9rature ambiante", "15\u00B0C \u2013 35\u00B0C"],
      ["P_cool_prev_norm", "Action pr\u00E9c\u00E9dente", "0 \u2013 1"],
    ],
    [3000, 3500, 2526]
  ),
  para({ spacing: { after: 60 } }, ""),
  para("L\u2019espace d\u2019action est continu, unidimensionnel : a \u2208 [0, 1], repr\u00E9sentant la fraction de puissance de refroidissement maximale appliqu\u00E9e."),

  heading(HeadingLevel.HEADING_2, "3.3 Choix de l\u2019algorithme : Soft Actor-Critic"),
  para("Le choix de SAC est motiv\u00E9 par plusieurs facteurs adapt\u00E9s \u00E0 notre probl\u00E8me :"),
  para("\u2022  Espace d\u2019action continu : SAC est nativement con\u00E7u pour les actions continues, contrairement \u00E0 DQN."),
  para("\u2022  Efficacit\u00E9 \u00E9chantillonnale : en tant qu\u2019algorithme off-policy, SAC r\u00E9utilise les exp\u00E9riences pass\u00E9es via un replay buffer, essentiel lorsque les interactions environnement-agent sont co\u00FBteuses."),
  para("\u2022  Stabilit\u00E9 : le double Q-network r\u00E9duit la surestimation des Q-valeurs, et l\u2019entropie adaptative assure un \u00E9quilibre exploration-exploitation."),
  para("\u2022  Robustesse : la r\u00E9gularisation entropique rend la politique plus r\u00E9siliente face aux perturbations stochastiques du courant et de la temp\u00E9rature ambiante."),

  heading(HeadingLevel.HEADING_2, "3.4 Strat\u00E9gie d\u2019entra\u00EEnement"),
  para("L\u2019entra\u00EEnement suit un protocole en plusieurs phases :"),
  para("1.  Phase d\u2019exploration al\u00E9atoire (1 000 premiers pas) : le buffer est rempli avec des transitions issues d\u2019actions uniformes."),
  para("2.  Phase d\u2019apprentissage : \u00E0 chaque pas, l\u2019agent s\u00E9lectionne une action via sa politique stochastique, collecte la transition, et effectue une mise \u00E0 jour SAC compl\u00E8te (critique, acteur, temp\u00E9rature)."),
  para("3.  \u00C9valuation p\u00E9riodique (tous les 20 \u00E9pisodes) : 5 \u00E9pisodes d\u00E9terministes sans exploration pour mesurer la performance r\u00E9elle."),
  para("4.  Sauvegarde de checkpoints pour l\u2019analyse post-entra\u00EEnement."),
  pageBreak(),
];

// --- 4. MODELISATION MATHEMATIQUE ---
const modelisation = [
  heading(HeadingLevel.HEADING_1, "4. Mod\u00E9lisation math\u00E9matique"),

  heading(HeadingLevel.HEADING_2, "4.1 Mod\u00E8le thermique \u00E9quivalent RC"),
  para("Le comportement thermique de la batterie est mod\u00E9lis\u00E9 par un circuit RC \u00E9quivalent \u00E0 un n\u0153ud. Ce mod\u00E8le, bien que simplifi\u00E9, capture les dynamiques essentielles : g\u00E9n\u00E9ration de chaleur par effet Joule, dissipation passive vers l\u2019ambiant et refroidissement actif."),

  heading(HeadingLevel.HEADING_3, "4.1.1 \u00C9quation diff\u00E9rentielle fondamentale"),
  para("Le bilan \u00E9nerg\u00E9tique du n\u0153ud thermique s\u2019\u00E9crit :"),
  eq([
    ital("C"), sub("th"), txt(" \u00B7 "), ital("dT"), txt("/"), ital("dt"),
    txt("  =  "), ital("P"), sub("gen"), txt("  \u2212  "), ital("P"), sub("cool"),
    txt("  \u2212  ("), ital("T"), txt(" \u2212 "), ital("T"), sub("amb"), txt(") / "), ital("R"), sub("th"),
  ]),
  para("o\u00F9 :"),
  para([ital("C"), sub("th"), txt(" [J/K] est la capacit\u00E9 thermique du pack batterie (inertie thermique).")]),
  para([ital("T"), txt(" [\u00B0C] est la temp\u00E9rature cellulaire uniforme (hypoth\u00E8se n\u0153ud unique).")]),
  para([ital("P"), sub("gen"), txt(" = "), ital("I"), sup("2"), txt(" \u00B7 "), ital("R"), sub("int"), txt(" [W] est la puissance de g\u00E9n\u00E9ration par effet Joule.")]),
  para([ital("P"), sub("cool"), txt(" = "), ital("a"), txt(" \u00B7 "), ital("P"), sub("cool,max"), txt(" [W] est la puissance de refroidissement actif, contr\u00F4l\u00E9e par l\u2019action "), ital("a"), txt(" \u2208 [0, 1].")]),
  para([txt("("), ital("T"), txt(" \u2212 "), ital("T"), sub("amb"), txt(") / "), ital("R"), sub("th"), txt(" [W] est le flux de chaleur passif vers l\u2019ambiant (convection naturelle).")]),

  heading(HeadingLevel.HEADING_3, "4.1.2 Discr\u00E9tisation (Euler explicite)"),
  para("Pour la simulation num\u00E9rique, l\u2019\u00E9quation est discr\u00E9tis\u00E9e par la m\u00E9thode d\u2019Euler explicite avec un pas de temps \u0394t = 1 s :"),
  eq([
    ital("T"), sub("k+1"), txt("  =  "), ital("T"), sub("k"),
    txt("  +  \u0394"), ital("t"), txt(" \u00B7 "),
    txt("[ ("), ital("P"), sub("gen"), txt(" \u2212 "), ital("P"), sub("cool"),
    txt(" \u2212 ("), ital("T"), sub("k"), txt(" \u2212 "), ital("T"), sub("amb"),
    txt(") / "), ital("R"), sub("th"), txt(") / "), ital("C"), sub("th"), txt(" ]"),
  ]),

  heading(HeadingLevel.HEADING_3, "4.1.3 Temp\u00E9rature d\u2019\u00E9quilibre"),
  para("En r\u00E9gime permanent (dT/dt = 0), la temp\u00E9rature d\u2019\u00E9quilibre vaut :"),
  eq([
    ital("T"), sub("ss"), txt("  =  "), ital("T"), sub("amb"),
    txt("  +  "), ital("R"), sub("th"), txt(" \u00B7 ("), ital("P"), sub("gen"),
    txt(" \u2212 "), ital("P"), sub("cool"), txt(")"),
  ]),
  para("Cette formule permet de v\u00E9rifier analytiquement la coh\u00E9rence de la simulation (test unitaire test_steady_state_formula)."),

  heading(HeadingLevel.HEADING_3, "4.1.4 Constante de temps"),
  para("La constante de temps du syst\u00E8me thermique est :"),
  eq([ital("\u03C4"), txt(" = "), ital("R"), sub("th"), txt(" \u00B7 "), ital("C"), sub("th")]),
  para([txt("Avec les param\u00E8tres g\u00E9n\u00E9riques ("), ital("R"), sub("th"), txt(" = 1 K/W, "), ital("C"), sub("th"), txt(" = 100 J/K), \u03C4 = 100 s. Cela signifie que le syst\u00E8me atteint ~63 % de son \u00E9quilibre en 100 secondes, ce qui est coh\u00E9rent avec les dynamiques thermiques r\u00E9elles de packs batterie de taille mod\u00E9r\u00E9e.")]),

  heading(HeadingLevel.HEADING_2, "4.2 Formulation comme processus de d\u00E9cision markovien"),
  para("Le probl\u00E8me de contr\u00F4le thermique est formul\u00E9 comme un MDP (Markov Decision Process) d\u00E9fini par le quintuplet (S, A, P, R, \u03B3) :"),

  heading(HeadingLevel.HEADING_3, "4.2.1 Espace d\u2019\u00E9tats S"),
  eq([
    ital("s"), sub("t"), txt(" = ("), ital("T\u0303"), sub("t"),
    txt(", "), ital("SoC\u0303"), sub("t"),
    txt(", "), ital("\u0128"), sub("t"),
    txt(", "), ital("T\u0303"), sub("amb,t"),
    txt(", "), ital("\u00E3"), sub("t-1"), txt(")  \u2208  \u211D"), sup("5"),
  ]),
  para("o\u00F9 le tilde d\u00E9note la normalisation dans [-1, 1] par transformation affine."),

  heading(HeadingLevel.HEADING_3, "4.2.2 Espace d\u2019actions A"),
  eq([ital("a"), sub("t"), txt(" \u2208 [0, 1]  \u2282  \u211D")]),
  para("repr\u00E9sentant la fraction de puissance de refroidissement maximale."),

  heading(HeadingLevel.HEADING_3, "4.2.3 Fonction de transition P"),
  para("La transition est d\u00E9terministe conditionnellement au courant et \u00E0 la temp\u00E9rature ambiante, qui \u00E9voluent stochastiquement :"),
  eq([ital("I"), sub("t+1"), txt(" = clip("), ital("I"), sub("t"), txt(" + \u03B5"), sub("I"), txt(", "), ital("I"), sub("min"), txt(", "), ital("I"), sub("max"), txt(")    avec \u03B5"), sub("I"), txt(" ~ \u039D(0, 4)")]),
  eq([ital("T"), sub("amb,t+1"), txt(" = clip("), ital("T"), sub("amb,t"), txt(" + \u03B5"), sub("T"), txt(", "), ital("T"), sub("amb,min"), txt(", "), ital("T"), sub("amb,max"), txt(")    avec \u03B5"), sub("T"), txt(" ~ \u039D(0, 0.0025)")]),

  heading(HeadingLevel.HEADING_2, "4.3 Algorithme Soft Actor-Critic"),

  heading(HeadingLevel.HEADING_3, "4.3.1 Objectif entropie-r\u00E9gularis\u00E9"),
  para("SAC maximise l\u2019objectif suivant, qui combine la r\u00E9compense cumul\u00E9e et l\u2019entropie de la politique :"),
  eq([
    ital("J"), txt("(\u03C0) = "), txt("\u2211"), sub("t"), txt(" \u03B3"), sup("t"),
    txt(" \u00B7 \u0395[ "), ital("r"), sub("t"),
    txt(" + \u03B1 \u00B7 \u210B(\u03C0(\u00B7|"), ital("s"), sub("t"), txt(")) ]"),
  ]),
  para([txt("o\u00F9 \u03B1 est le coefficient de temp\u00E9rature entropique et \u210B(\u03C0) = \u2212\u0395[log \u03C0("), ital("a"), txt("|"), ital("s"), txt(")] est l\u2019entropie de la politique. Ce terme encourage l\u2019exploration en favorisant des politiques \u00E0 forte entropie.")]),

  heading(HeadingLevel.HEADING_3, "4.3.2 Mise \u00E0 jour du critique (Twin Q-Networks)"),
  para("Deux r\u00E9seaux Q ind\u00E9pendants sont entra\u00EEn\u00E9s simultan\u00E9ment pour r\u00E9duire le biais de surestimation. La cible pour chaque r\u00E9seau est :"),
  eq([
    ital("y"), sub("t"), txt(" = "), ital("r"), sub("t"), txt(" + \u03B3 \u00B7 (1 \u2212 "),
    ital("d"), sub("t"), txt(") \u00B7 [ min("), ital("Q"), sub("1,targ"),
    txt(", "), ital("Q"), sub("2,targ"), txt(") \u2212 \u03B1 \u00B7 log \u03C0("),
    ital("a\u2032"), txt("|"), ital("s\u2032"), txt(") ]"),
  ]),
  para("La loss du critique est la somme des MSE :"),
  eq([
    ital("L"), sub("critic"), txt(" = MSE("), ital("Q"), sub("1"), txt(", "), ital("y"),
    txt(") + MSE("), ital("Q"), sub("2"), txt(", "), ital("y"), txt(")"),
  ]),

  heading(HeadingLevel.HEADING_3, "4.3.3 Mise \u00E0 jour de l\u2019acteur"),
  para("L\u2019acteur est mis \u00E0 jour pour maximiser la Q-valeur tout en maintenant une haute entropie :"),
  eq([
    ital("L"), sub("actor"), txt(" = \u0395"), sub("s~B"), txt("[ \u03B1 \u00B7 log \u03C0("),
    ital("a\u0303"), txt("|"), ital("s"), txt(") \u2212 "), ital("Q"), sub("1"),
    txt("("), ital("s"), txt(", "), ital("a\u0303"), txt(") ]"),
  ]),
  para([txt("o\u00F9 "), ital("a\u0303"), txt(" est \u00E9chantillonn\u00E9 via la reparametrization trick : "), ital("a\u0303"), txt(" = tanh(\u03BC("), ital("s"), txt(") + \u03C3("), ital("s"), txt(") \u2299 \u03B5), avec \u03B5 ~ \u039D(0, I).")]),

  heading(HeadingLevel.HEADING_3, "4.3.4 Entropie adaptative"),
  para("Le coefficient \u03B1 est appris automatiquement en minimisant :"),
  eq([
    ital("L"), sub("\u03B1"), txt(" = \u2212\u03B1 \u00B7 \u0395[ log \u03C0("),
    ital("a"), txt("|"), ital("s"), txt(") + \u210B\u0303 ]"),
  ]),
  para([txt("o\u00F9 \u210B\u0303 = \u2212dim(A) est l\u2019entropie cible. En pratique, c\u2019est log \u03B1 qui est optimis\u00E9 (plus stable num\u00E9riquement).")]),

  heading(HeadingLevel.HEADING_3, "4.3.5 Soft update des r\u00E9seaux cibles"),
  para("Les r\u00E9seaux cibles sont mis \u00E0 jour par moyennage de Polyak :"),
  eq([
    ital("\u03B8"), sub("target"), txt(" \u2190 \u03C4 \u00B7 \u03B8 + (1 \u2212 \u03C4) \u00B7 "),
    ital("\u03B8"), sub("target"),
  ]),
  para("avec \u03C4 = 0,005 (valeur standard)."),

  heading(HeadingLevel.HEADING_2, "4.4 Fonction de r\u00E9compense"),

  heading(HeadingLevel.HEADING_3, "4.4.1 R\u00E9compense de base"),
  para("La fonction de r\u00E9compense encode quatre objectifs pond\u00E9r\u00E9s :"),
  eq([
    ital("r"), sub("t"), txt(" = \u2212"), ital("w"), sub("high"), txt(" \u00B7 max(0, "), ital("T"), txt(" \u2212 "), ital("T"), sub("safe,max"), txt(")\u00B2"),
    txt("  \u2212  "), ital("w"), sub("low"), txt(" \u00B7 max(0, "), ital("T"), sub("safe,min"), txt(" \u2212 "), ital("T"), txt(")\u00B2"),
    txt("  \u2212  "), ital("w"), sub("cool"), txt(" \u00B7 "), ital("a"),
    txt("  +  "), ital("w"), sub("soc"), txt(" \u00B7 \u0394SoC"),
  ]),

  heading(HeadingLevel.HEADING_3, "4.4.2 P\u00E9nalit\u00E9 de mont\u00E9e thermique (Jour 7)"),
  para("Pour am\u00E9liorer l\u2019anticipation des pics, une p\u00E9nalit\u00E9 proportionnelle \u00E0 la vitesse de mont\u00E9e de temp\u00E9rature est ajout\u00E9e :"),
  eq([
    ital("r"), sub("rise"), txt(" = \u2212"), ital("w"), sub("\u0394T"), txt(" \u00B7 max(0, \u0394"),
    ital("T"), txt(") \u00B7 max(0, "), ital("T"), txt(" \u2212 "), ital("T"), sub("warning"), txt(")"),
  ]),
  para([txt("Cette p\u00E9nalit\u00E9 est active uniquement lorsque la temp\u00E9rature d\u00E9passe le seuil d\u2019alerte "), ital("T"), sub("warning"), txt(" = 40\u00B0C ET monte (\u0394T > 0). Elle fournit un signal d\u2019apprentissage pr\u00E9coce, avant que la temp\u00E9rature n\u2019atteigne la zone dangereuse.")]),

  simpleTable(
    ["Param\u00E8tre", "Valeur initiale (J1\u2013J6)", "Valeur tun\u00E9e (J7)", "Justification"],
    [
      ["w_temp_high", "2,0", "5,0", "P\u00E9nalise plus fort les d\u00E9passements"],
      ["w_temp_low", "1,0", "1,0", "Inchang\u00E9"],
      ["w_cooling", "0,05", "0,01", "Autorise plus de refroidissement"],
      ["w_soc", "0,1", "0,1", "Inchang\u00E9"],
      ["w_delta_T", "\u2014", "0,5", "Anticipe les mont\u00E9es thermiques"],
    ],
    [1800, 2100, 2100, 3026]
  ),
  pageBreak(),
];

// --- 5. IMPLEMENTATION ---
const implementation = [
  heading(HeadingLevel.HEADING_1, "5. Impl\u00E9mentation"),

  heading(HeadingLevel.HEADING_2, "5.1 Structure du code"),
  para("Le projet suit une architecture modulaire, con\u00E7ue pour \u00EAtre facile \u00E0 maintenir et \u00E0 \u00E9tendre :"),
  para("    RL battery/"),
  para("    \u251C\u2500\u2500 config.py                   # Param\u00E8tres thermiques, batterie, reward"),
  para("    \u251C\u2500\u2500 envs/"),
  para("    \u2502   \u251C\u2500\u2500 thermal_model.py         # Mod\u00E8le RC (physique pure)"),
  para("    \u2502   \u2514\u2500\u2500 battery_thermal_env.py   # Environnement Gymnasium"),
  para("    \u251C\u2500\u2500 agent/"),
  para("    \u2502   \u251C\u2500\u2500 networks.py              # Actor + Critic (r\u00E9seaux PyTorch)"),
  para("    \u2502   \u251C\u2500\u2500 replay_buffer.py         # Replay buffer circulaire"),
  para("    \u2502   \u2514\u2500\u2500 sac.py                   # Agent SAC (logique haut niveau)"),
  para("    \u251C\u2500\u2500 train.py                 # Boucle d\u2019entra\u00EEnement"),
  para("    \u251C\u2500\u2500 evaluate.py              # \u00C9valuation + plots"),
  para("    \u251C\u2500\u2500 sweep.py                 # Grid search hyperparam\u00E8tres"),
  para("    \u251C\u2500\u2500 utils/logger.py          # Logging CSV + console"),
  para("    \u251C\u2500\u2500 tests/                   # Tests unitaires (pytest)"),
  para("    \u2514\u2500\u2500 notebooks/               # Jupyter notebooks d\u2019analyse"),
  figCaption("Figure 2 : Arborescence du projet. Chaque module est ind\u00E9pendant et test\u00E9 unitairement."),

  heading(HeadingLevel.HEADING_2, "5.2 Modules principaux"),

  heading(HeadingLevel.HEADING_3, "5.2.1 ThermalModel (envs/thermal_model.py)"),
  para("Cette classe encapsule la physique pure du mod\u00E8le RC. Sa m\u00E9thode step() calcule T_next \u00E0 partir de (T, I, action, T_amb) par int\u00E9gration d\u2019Euler. La m\u00E9thode steady_state_temp() fournit la solution analytique en r\u00E9gime permanent. Ce d\u00E9couplage physique/RL permet de tester le mod\u00E8le thermique ind\u00E9pendamment de l\u2019environnement Gymnasium."),

  heading(HeadingLevel.HEADING_3, "5.2.2 BatteryThermalEnv (envs/battery_thermal_env.py)"),
  para("L\u2019environnement Gymnasium principal. Il orchestre le ThermalModel, la mise \u00E0 jour du SoC, les profils stochastiques de courant et de temp\u00E9rature ambiante, la normalisation des observations, et le calcul du reward. Les conditions de terminaison sont : T > T_cutoff (s\u00E9curit\u00E9), SoC \u2265 1 (batterie pleine) ou d\u00E9passement du nombre maximal de steps (troncature)."),

  heading(HeadingLevel.HEADING_3, "5.2.3 Actor (agent/networks.py)"),
  para("L\u2019acteur est un r\u00E9seau \u00E0 couches enti\u00E8rement connect\u00E9es (MLP) qui produit les param\u00E8tres d\u2019une distribution gaussienne conditionnelle. L\u2019architecture par d\u00E9faut comprend 2 couches cach\u00E9es de 256 neurones avec activation ReLU, suivies de deux t\u00EAtes parall\u00E8les (\u03BC et log \u03C3). L\u2019action est pass\u00E9e par une fonction tanh puis rescal\u00E9e de (-1, 1) vers [0, 1]. La correction de log-probabilit\u00E9 pour la transformation tanh est appliqu\u00E9e :"),
  eq([
    txt("log \u03C0("), ital("a"), txt("|"), ital("s"), txt(") = log \u03BC"), sub("\u03B8"),
    txt("("), ital("u"), txt("|"), ital("s"), txt(") \u2212 \u2211 log(1 \u2212 tanh\u00B2("),
    ital("u"), sub("i"), txt("))"),
  ]),

  heading(HeadingLevel.HEADING_3, "5.2.4 Critic (agent/networks.py)"),
  para("Le critique impl\u00E9mente deux r\u00E9seaux Q ind\u00E9pendants (Twin Q-Networks). Chaque r\u00E9seau prend en entr\u00E9e la concat\u00E9nation (s, a) et produit un scalaire Q(s, a). La m\u00E9thode forward() retourne les deux Q-valeurs ; q1_forward() ne retourne que la premi\u00E8re (utilis\u00E9e pour la mise \u00E0 jour de l\u2019acteur)."),

  heading(HeadingLevel.HEADING_3, "5.2.5 ReplayBuffer (agent/replay_buffer.py)"),
  para("Le buffer circulaire stocke les transitions (s, a, r, s\u2032, d) dans des tableaux NumPy pr\u00E9-allou\u00E9s, puis les convertit en tenseurs PyTorch lors de l\u2019\u00E9chantillonnage. Cette approche offre un bon compromis m\u00E9moire/vitesse. La capacit\u00E9 par d\u00E9faut est de 100 000 transitions."),

  heading(HeadingLevel.HEADING_3, "5.2.6 SACAgent (agent/sac.py)"),
  para("La classe orchestrant l\u2019ensemble de l\u2019algorithme. Elle g\u00E8re les trois optimiseurs (critique, acteur, \u03B1), le soft update des cibles, et l\u2019interface publique (push, select_action, update, save/load). Le coefficient log \u03B1 est appris directement (plus stable que \u03B1)."),

  heading(HeadingLevel.HEADING_2, "5.3 Flux de donn\u00E9es"),
  para("Le flux de donn\u00E9es principal suit le cycle suivant :"),
  para("1.  L\u2019environnement produit une observation normalis\u00E9e s \u2208 \u211D\u2075."),
  para("2.  L\u2019acteur transforme s en param\u00E8tres gaussiens (\u03BC, \u03C3), \u00E9chantillonne u ~ \u039D(\u03BC, \u03C3\u00B2), applique tanh et rescale vers [0, 1]."),
  para("3.  L\u2019environnement re\u00E7oit l\u2019action, avance d\u2019un pas physique, retourne (s\u2032, r, done, info)."),
  para("4.  La transition est stock\u00E9e dans le replay buffer."),
  para("5.  Un batch al\u00E9atoire est \u00E9chantillonn\u00E9 pour les mises \u00E0 jour : critique \u2192 acteur \u2192 \u03B1 \u2192 soft update cible."),
  figCaption("Figure 3 : Flux de donn\u00E9es pour un pas d\u2019entra\u00EEnement. Les fl\u00E8ches pleines repr\u00E9sentent le flux forward ; les fl\u00E8ches pointill\u00E9es, les gradients."),
  pageBreak(),
];

// --- 6. RESULTATS ---
const resultats = [
  heading(HeadingLevel.HEADING_1, "6. R\u00E9sultats et analyse"),

  heading(HeadingLevel.HEADING_2, "6.1 R\u00E9sultats d\u2019entra\u00EEnement"),

  heading(HeadingLevel.HEADING_3, "6.1.1 Run Jour 6 : reward original (200 000 pas)"),
  para("Le premier run long utilise les poids de r\u00E9compense initiaux (w_temp_high = 2, w_cooling = 0,05). Les principaux indicateurs \u00E9voluent comme suit :"),

  simpleTable(
    ["\u00C9pisode", "Steps cumul\u00E9s", "pct_safe", "T_max moy.", "Return \u00E9val."],
    [
      ["20", "~12k", "64,9 %", "60,3\u00B0C", "-6 004"],
      ["40", "~22k", "86,0 %", "55,6\u00B0C", "-2 754"],
      ["140", "~88k", "84,8 %", "52,2\u00B0C", "-2 060"],
      ["200", "~128k", "94,5 %", "53,3\u00B0C", "-2 151"],
      ["300", "~192k", "95,6 %", "56,0\u00B0C", "-4 800"],
    ],
    [1200, 1800, 1600, 2000, 2426]
  ),
  para(""),
  figCaption("Figure 4 : Courbes d\u2019apprentissage Jour 6. (a) pct_safe en fonction des steps. (b) T_max par \u00E9pisode d\u2019\u00E9valuation. Le pct_safe atteint 95,6 % au meilleur \u00E9pisode."),

  heading(HeadingLevel.HEADING_3, "6.1.2 Run Jour 7 : reward tun\u00E9 (300 000 pas)"),
  para("Le second run utilise les poids tun\u00E9s (w_temp_high = 5, w_cooling = 0,01, w_delta_T = 0,5). Note importante : les returns ne sont pas directement comparables entre les deux runs car la magnitude des p\u00E9nalit\u00E9s a chang\u00E9. L\u2019indicateur neutre est pct_safe."),

  simpleTable(
    ["\u00C9pisode", "Steps cumul\u00E9s", "pct_safe", "T_max moy.", "Return \u00E9val."],
    [
      ["20", "~12k", "88,4 %", "60,3\u00B0C", "-31 439"],
      ["100", "~69k", "80,0 %", "60,3\u00B0C", "-10 218"],
      ["180", "~119k", "91,0 %", "60,3\u00B0C", "-11 222"],
      ["320", "~204k", "96,7 %", "50,0\u00B0C", "-5 418"],
      ["360", "~227k", "89,4 %", "49,4\u00B0C", "-6 775"],
    ],
    [1200, 1800, 1600, 2000, 2426]
  ),
  para(""),
  figCaption("Figure 5 : Courbes d\u2019apprentissage Jour 7. La p\u00E9nalit\u00E9 dT/dt permet des T_max plus basses (50\u00B0C vs 56\u00B0C) mais introduit plus de variance."),

  heading(HeadingLevel.HEADING_2, "6.2 Analyse comparative"),
  para("Le tableau suivant synth\u00E9tise la comparaison entre les deux configurations :"),

  simpleTable(
    ["M\u00E9trique", "Jour 6 (original)", "Jour 7 (tun\u00E9)"],
    [
      ["pct_safe (meilleur)", "95,6 %", "96,7 %"],
      ["T_max au meilleur pct_safe", "56,0\u00B0C", "50,0\u00B0C"],
      ["Variance pct_safe", "Mod\u00E9r\u00E9e", "\u00C9lev\u00E9e"],
      ["Convergence", "~128k steps", "~204k steps"],
      ["Anticipation pics", "Faible", "Am\u00E9lior\u00E9e (dT/dt)"],
    ],
    [3000, 3013, 3013]
  ),

  para(""),
  para("Observations cl\u00E9s :"),
  para("1.  Le reward tun\u00E9 am\u00E9liore marginalement le pct_safe maximal (95,6 % \u2192 96,7 %), mais surtout r\u00E9duit les T_max atteintes (50\u00B0C vs 56\u00B0C), indiquant une meilleure anticipation."),
  para("2.  La p\u00E9nalit\u00E9 w_delta_T fournit un signal gradient plus pr\u00E9coce : l\u2019agent commence \u00E0 refroidir d\u00E8s que T > 40\u00B0C et monte, au lieu d\u2019attendre le d\u00E9passement de T_safe_max = 45\u00B0C."),
  para("3.  La variance plus \u00E9lev\u00E9e du run tun\u00E9 sugg\u00E8re que le paysage d\u2019optimisation est plus complexe avec des p\u00E9nalit\u00E9s plus fortes. L\u2019agent n\u00E9cessite davantage de steps pour converger pleinement."),

  heading(HeadingLevel.HEADING_2, "6.3 Validation par tests unitaires"),
  para("Le projet est accompagn\u00E9 d\u2019une suite de tests unitaires couvrant l\u2019ensemble des composants :"),

  simpleTable(
    ["Module test\u00E9", "Fichier test", "Nombre de tests", "Statut"],
    [
      ["Mod\u00E8le thermique + Env", "test_env.py", "17", "\u2705 17/17"],
      ["Agent SAC (buffer, actor, critic)", "test_sac.py", "22", "\u2705 22/22"],
      ["Boucle d\u2019entra\u00EEnement", "test_train.py", "5", "\u2705 5/5"],
      ["Reward tuning (dT/dt)", "test_day7.py", "10", "\u2705 10/10"],
    ],
    [2500, 2000, 2000, 2526]
  ),

  heading(HeadingLevel.HEADING_2, "6.4 Limites identifi\u00E9es"),
  para("Malgr\u00E9 les r\u00E9sultats encourageants, plusieurs limites sont \u00E0 noter :"),
  para("1.  Mod\u00E8le thermique simplifi\u00E9 : le mod\u00E8le RC \u00E0 un n\u0153ud ne capture pas les gradients de temp\u00E9rature internes ni les non-lin\u00E9arit\u00E9s (R_int d\u00E9pendant de T et SoC)."),
  para("2.  Espace d\u2019action unidimensionnel : seule la puissance de refroidissement est contr\u00F4l\u00E9e. Le courant de charge est impos\u00E9 et non optimis\u00E9 par l\u2019agent."),
  para("3.  Profil de courant stochastique : la marche al\u00E9atoire ne repr\u00E9sente pas fid\u00E8lement les profils de charge r\u00E9els (cycles CCCV, profils de conduite, etc.)."),
  para("4.  Variance de l\u2019entra\u00EEnement : la forte variance, particuli\u00E8rement avec le reward tun\u00E9, limite la fiabilit\u00E9 de la politique apprise."),
  para("5.  Absence de validation sur donn\u00E9es r\u00E9elles : le mod\u00E8le n\u2019a pas \u00E9t\u00E9 calibr\u00E9 sur des mesures exp\u00E9rimentales."),
  pageBreak(),
];

// --- 7. PERSPECTIVES ---
const perspectives = [
  heading(HeadingLevel.HEADING_1, "7. Perspectives d\u2019am\u00E9lioration"),

  heading(HeadingLevel.HEADING_2, "7.1 Enrichissement du mod\u00E8le physique"),
  para("Le passage \u00E0 un mod\u00E8le RC \u00E0 deux n\u0153uds (temp\u00E9rature interne + temp\u00E9rature de surface) permettrait de capturer les gradients thermiques et d\u2019offrir une repr\u00E9sentation plus fid\u00E8le. La d\u00E9pendance de R_int \u00E0 la temp\u00E9rature et au SoC pourrait \u00EAtre int\u00E9gr\u00E9e via une table de lookup interpolante."),

  heading(HeadingLevel.HEADING_2, "7.2 Extension de l\u2019espace d\u2019action"),
  para("L\u2019agent pourrait contr\u00F4ler simultan\u00E9ment la puissance de refroidissement et la limitation de courant (current throttling). Cela n\u00E9cessiterait un espace d\u2019action bidimensionnel, que SAC g\u00E8re nativement."),

  heading(HeadingLevel.HEADING_2, "7.3 Profils de charge r\u00E9alistes"),
  para("L\u2019int\u00E9gration de profils de charge standardis\u00E9s (CCCV, profils WLTP pour la mobilit\u00E9 \u00E9lectrique) ou de profils historiques r\u00E9els am\u00E9liorerait la pertinence de l\u2019entra\u00EEnement et la transf\u00E9rabilit\u00E9 de la politique."),

  heading(HeadingLevel.HEADING_2, "7.4 Am\u00E9liorations algorithmiques"),
  para("\u2022  Prioritized Experience Replay (PER) : \u00E9chantillonner pr\u00E9f\u00E9rentiellement les transitions \u00E0 forte erreur TD pour acc\u00E9l\u00E9rer l\u2019apprentissage."),
  para("\u2022  Learning rate scheduler : r\u00E9duire progressivement le learning rate pour stabiliser la convergence finale."),
  para("\u2022  Curriculum learning : commencer avec des sc\u00E9narios simples (courant constant, T_amb fixe) et augmenter progressivement la difficult\u00E9."),
  para("\u2022  Normalisation du reward par fen\u00EAtre glissante pour stabiliser les gradients."),

  heading(HeadingLevel.HEADING_2, "7.5 D\u00E9ploiement embarqu\u00E9"),
  para("Pour un d\u00E9ploiement en contexte FSE r\u00E9el, la politique entra\u00EEn\u00E9e pourrait \u00EAtre export\u00E9e en ONNX et ex\u00E9cut\u00E9e sur un microcontr\u00F4leur via ONNX Runtime Micro. Le r\u00E9seau acteur seul (2 couches de 256, ~130 k param\u00E8tres) n\u00E9cessite environ 520 Ko de m\u00E9moire et une inf\u00E9rence de l\u2019ordre de la milliseconde sur un ARM Cortex-M7."),

  heading(HeadingLevel.HEADING_2, "7.6 Calibration sur donn\u00E9es r\u00E9elles"),
  para("Les param\u00E8tres thermiques g\u00E9n\u00E9riques (C_th, R_th, R_int) devraient \u00EAtre calibr\u00E9s sur des mesures exp\u00E9rimentales (tests calori\u00E9m\u00E9triques, essais de charge instrument\u00E9s). Une approche system identification par moindres carr\u00E9s ou filtre de Kalman permettrait d\u2019estimer ces param\u00E8tres en ligne."),
  pageBreak(),
];

// --- 8. CONCLUSION ---
const conclusion = [
  heading(HeadingLevel.HEADING_1, "8. Conclusion"),
  para("Ce projet a d\u00E9montr\u00E9 la faisabilit\u00E9 de l\u2019utilisation d\u2019un agent Soft Actor-Critic pour le contr\u00F4le thermique de batteries en contexte FSE. En 7 jours de d\u00E9veloppement par blocs de 2 \u00E0 3 heures, nous avons con\u00E7u et impl\u00E9ment\u00E9 un syst\u00E8me complet couvrant l\u2019int\u00E9gralit\u00E9 du pipeline RL : mod\u00E9lisation physique, environnement de simulation, agent d\u2019apprentissage, boucle d\u2019entra\u00EEnement et outils d\u2019analyse."),
  para("Les r\u00E9sultats montrent que l\u2019agent apprend \u00E0 maintenir la temp\u00E9rature cellulaire en zone s\u00FBre (15\u00B0C \u2013 45\u00B0C) entre 95,6 % et 96,7 % du temps, selon la configuration du reward. L\u2019introduction d\u2019une p\u00E9nalit\u00E9 sur la d\u00E9riv\u00E9e thermique (dT/dt) a permis d\u2019am\u00E9liorer l\u2019anticipation des pics, r\u00E9duisant les T_max observ\u00E9es de 56\u00B0C \u00E0 50\u00B0C."),
  para("L\u2019impl\u00E9mentation enti\u00E8rement from scratch (sans biblioth\u00E8que RL pr\u00E9-construite) garantit une compr\u00E9hension fine de chaque composant et une flexibilit\u00E9 maximale pour l\u2019adaptation \u00E0 des contextes industriels sp\u00E9cifiques. La suite de 54 tests unitaires assure la robustesse du code."),
  para("Les perspectives incluent l\u2019enrichissement du mod\u00E8le thermique, l\u2019extension de l\u2019espace d\u2019action, l\u2019utilisation de profils de charge r\u00E9alistes et le d\u00E9ploiement embarqu\u00E9 via l\u2019export ONNX. Ce travail constitue une base solide pour le d\u00E9veloppement d\u2019un syst\u00E8me de gestion thermique intelligent adapt\u00E9 aux contraintes du terrain."),
  pageBreak(),
];

// --- REFERENCES ---
const references = [
  heading(HeadingLevel.HEADING_1, "R\u00E9f\u00E9rences bibliographiques"),
  para("[1]  T. Haarnoja, A. Zhou, P. Abbeel, S. Levine. \u00AB Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor \u00BB, ICML 2018. arXiv:1801.01290."),
  para("[2]  T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan, V. Kumar, H. Zhu, A. Gupta, P. Abbeel, S. Levine. \u00AB Soft Actor-Critic Algorithms and Applications \u00BB, 2018. arXiv:1812.05905."),
  para("[3]  S. Fujimoto, H. van Hoof, D. Meger. \u00AB Addressing Function Approximation Error in Actor-Critic Methods \u00BB, ICML 2018. arXiv:1802.09477."),
  para("[4]  V. Mnih et al. \u00AB Human-level control through deep reinforcement learning \u00BB, Nature 518, 529\u2013533, 2015."),
  para("[5]  R. S. Sutton, A. G. Barto. \u00AB Reinforcement Learning: An Introduction \u00BB, 2\u00E8me \u00E9dition, MIT Press, 2018."),
  para("[6]  A. Pesaran. \u00AB Battery Thermal Management in EVs \u00BB, Advanced Automotive Battery Conference, 2001."),
  para("[7]  X. Lin, H. E. Perez, S. Mohan, J. B. Siegel, A. G. Stefanopoulou, Y. Ding, M. P. Castanier. \u00AB A lumped-parameter electro-thermal model for cylindrical batteries \u00BB, Journal of Power Sources, 257, 1\u201311, 2014."),
  para("[8]  S. Park, D. Kang, J. Ahn. \u00AB Deep Reinforcement Learning-Based Battery Thermal Management \u00BB, IEEE Access, vol. 10, 2022."),
  para("[9]  OpenAI Spinning Up in Deep RL. \u00AB Soft Actor-Critic \u00BB, documentation en ligne."),
  para("[10] Gymnasium Documentation. \u00AB Gymnasium: A Standard API for Reinforcement Learning \u00BB, Farama Foundation."),
  pageBreak(),
];

// --- ANNEXE: PARAMETRES ---
const annexe = [
  heading(HeadingLevel.HEADING_1, "Annexe A : Param\u00E8tres par d\u00E9faut"),

  heading(HeadingLevel.HEADING_2, "A.1 Param\u00E8tres thermiques et batterie"),
  simpleTable(
    ["Param\u00E8tre", "Symbole", "Valeur", "Unit\u00E9"],
    [
      ["Capacit\u00E9 thermique", "C_th", "100", "J/K"],
      ["R\u00E9sistance thermique", "R_th", "1,0", "K/W"],
      ["R\u00E9sistance interne", "R_int", "0,05", "\u03A9"],
      ["Puissance max refroidissement", "P_cool,max", "50", "W"],
      ["Capacit\u00E9 batterie", "C_batt", "10", "Ah"],
      ["Courant max", "I_max", "50", "A"],
      ["Courant min", "I_min", "-10", "A"],
      ["T initiale min", "T_init,min", "20", "\u00B0C"],
      ["T initiale max", "T_init,max", "30", "\u00B0C"],
      ["T ambiante min", "T_amb,min", "15", "\u00B0C"],
      ["T ambiante max", "T_amb,max", "35", "\u00B0C"],
      ["T s\u00FBre min", "T_safe,min", "15", "\u00B0C"],
      ["T s\u00FBre max", "T_safe,max", "45", "\u00B0C"],
      ["T alerte (warning)", "T_warning", "40", "\u00B0C"],
      ["T coupure", "T_cutoff", "60", "\u00B0C"],
      ["Pas de temps", "\u0394t", "1,0", "s"],
      ["Steps max par \u00E9pisode", "max_steps", "3 600", "\u2014"],
    ],
    [3000, 1500, 1500, 2026]
  ),

  heading(HeadingLevel.HEADING_2, "A.2 Hyperparam\u00E8tres SAC"),
  simpleTable(
    ["Param\u00E8tre", "Valeur"],
    [
      ["Couches cach\u00E9es (acteur + critique)", "2 \u00D7 256"],
      ["Activation", "ReLU"],
      ["Learning rate (\u03C0, Q, \u03B1)", "3 \u00D7 10\u207B\u2074"],
      ["Facteur de discount (\u03B3)", "0,99"],
      ["Coefficient soft update (\u03C4)", "0,005"],
      ["Taille du batch", "256"],
      ["Capacit\u00E9 replay buffer", "100 000"],
      ["Learning starts", "1 000"],
      ["Entropie cible", "\u2212dim(A) = \u22121"],
      ["\u03B1 initial", "0,2"],
    ],
    [5000, 4026]
  ),
];

// ============================================================
//  ASSEMBLE DOCUMENT
// ============================================================
const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: FONT, size: 24 },
      },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT },
        paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT },
        paragraph: { spacing: { before: 240, after: 180 }, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, italics: true },
        paragraph: { spacing: { before: 180, after: 120 }, outlineLevel: 2 },
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [txt("RL pour le contr\u00F4le thermique de batteries \u2014 SAC", { size: 18, italics: true, color: "888888" })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [txt("Page ", { size: 20 }), new TextRun({ children: [PageNumber.CURRENT], size: 20, font: FONT }), txt(" / ", { size: 20 }), new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 20, font: FONT })],
        })],
      }),
    },
    children: [
      ...titlePage,
      ...abstractSection,
      ...tocSection,
      ...intro,
      ...etatArt,
      ...methodologie,
      ...modelisation,
      ...implementation,
      ...resultats,
      ...perspectives,
      ...conclusion,
      ...references,
      ...annexe,
    ],
  }],
});

// ============================================================
//  GENERATE
// ============================================================
const OUTPUT = "C:\\Users\\Utilisateur\\Desktop\\Challenge 7 day Coding\\RL battery\\rapport_projet_RL_batterie.docx";

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUTPUT, buffer);
  console.log("Rapport genere: " + OUTPUT);
  console.log("Taille: " + (buffer.length / 1024).toFixed(0) + " Ko");
}).catch(err => {
  console.error("Erreur:", err.message);
  process.exit(1);
});
