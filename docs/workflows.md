# Choose the control you need

[Home](../README.md) · [Rhyme profiles](rhyme-rules.md)

Rhyme tools already offer near rhymes, syllable counts, phrase search, and graded similarity. Those are useful foundations. The question for a writing session is where you want to make the decision: while browsing words, developing a thought, or comparing actual lines in your delivery.

The table compares **workflows**, rather than auditing named products. Individual tools can span several columns. It is not a claim that every other tool lacks a feature, or a benchmark of lyric quality.

| Writing need | Dictionary-led exploration | Conversational writing | Mumble Rhyme's explicit control |
| --- | --- | --- | --- |
| Find a word family | Browse sounds, spellings, syllables, and phrases | Ask for vocabulary and associations | Search installed English readings by soft phonetic weight |
| Turn a feeling into a line | Build the line around a discovered word | Discuss meaning and generate rewrites | Generate English options from a brief, including a Chinese brief, then expose component scores |
| Try a sound that works when sung | Listen through a list of near matches | Describe the delivery in a prompt | Store a direct vowel-pair weight and a sung pronunciation in YAML |
| Compare your own alternatives | Cross-check each ending | Ask for an editorial preference | Score every candidate with the same pinned model and profile, including left/right context |
| Keep a risky but meaningful ending | Save it while exploring | Ask for more adventurous options | Retain every supplied candidate with a positive rhyme weight and disclose unknown readings |
| Reuse decisions in another session | Return to favorite word families | Reuse a brief or conversation | Save a profile or send the same JSON through Python, CLI, or MCP |

## “These sounds work for this line”

A global “more near rhymes” setting changes the breadth of a search. A pair rule says something narrower: **for this profile, this vowel against that vowel is acceptable at this weight**. You can make it directional or restrict it to the last syllable without creating transitive equivalence groups.

This is useful for keeping delivery choices explicit. It does not demonstrate that a machine can hear your performance: you provide the pronunciation assumptions yourself.

## “I already have three ways to say it”

Bring the phrases rather than restarting from a word list. The model can score the continuation in context while the phonetic layer explains the ending. A candidate with a weaker rhyme may still win on language fit. Increasing rhyme strength makes the tradeoff visible.

A stronger external assistant can do the editorial work—meaning, idiom, narrative, alternatives—and use this local tool for repeatable candidate comparisons. Tool scores do not replace its judgment or yours.

## “Will it make better songs?”

That needs a listening and writing evaluation. Current evidence demonstrates working generation, scoring, custom rules, and tool transport on one measured local setup. It does not demonstrate superiority over a dictionary, another model, or a human collaborator.

Use dictionary exploration for breadth, conversation for developing the thought, and explicit local ranking when you want to preserve and inspect your own rhyme tolerances.
