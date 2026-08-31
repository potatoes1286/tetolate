Tetolate includes open stand-in fonts for dialogue, thoughts, computer text,
handwriting, and sound effects. You do not need to add fonts before use.

To use your own fonts, place `.ttf`, `.otf`, `.ttc`, or `.otc` files here.
Copy `font_use.example.txt` to `font_use.txt`, then describe when the VLM should
use each installed font. A user `font_use.txt` replaces the bundled role list,
and a user font file takes precedence over a bundled file with the same name.

The contents of `font_use.txt` are provided to the VLM as-is, so natural
language descriptions are fine. User font files and `font_use.txt` are ignored
by Git.

My personal setup:

wild_words.ttf: talking, narrator, general text, fallback
wild_words_italic.ttf: thinking, monologue
roboto.ttf: computer text
architects_daughter.ttf: hand written text
orange_fizz_italic.ttf: sfx, loud shout

Obtain and license user fonts separately. If no user fonts are installed,
tetolate uses the bundled Comic Neue Bold backup and bundled role mapping.
