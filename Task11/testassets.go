// Package testassets embeds the files from testdata/ so the server can use
// them at runtime: preset dialogs for the chat-scenario blobs and quiz files
// for the "+" menu. The embedding package lives at the module root because
// the go:embed directive cannot reference files above or beside its own
// directory, and packages inside testdata/ are ignored by the go tool.
package testassets

import _ "embed"

// DialogMemoryCheck is the preset dialog exercising all three memory layers:
// explicit profile facts (remember tool), a task with notes (working memory)
// and facts repeated across turns so sticky-facts compression promotes them
// into the long-term layer after two confirmations.
//
//go:embed testdata/dialogs/memory-check.txt
var DialogMemoryCheck string

// QuizMemory is a Q&A quiz about the facts from DialogMemoryCheck, for
// checking how well the dialog layer retains them after compression.
//
//go:embed testdata/quizzes/memory-quiz.txt
var QuizMemory string
