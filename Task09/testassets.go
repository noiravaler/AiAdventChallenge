// Package testassets embeds the files from testdata/ so the server can use
// them at runtime: preset dialogs for the chat-scenario blobs and quiz files
// for the "+" menu. The embedding package lives at the module root because
// go:embed cannot reference files above or beside its own directory, and
// packages inside testdata/ are ignored by the go tool.
package testassets

import _ "embed"

// DialogCurtains is the preset sales dialog about curtains for a kindergarten
// ("Диалог про шторы"): alternating "User:" / "Model:" turns.
//
//go:embed testdata/dialogs/curtains-kindergarten-sale.txt
var DialogCurtains string

// QuizCurtains is a Q&A quiz ("Вопрос:" / "Ответ:") about the facts from
// DialogCurtains, for checking how well the model retains dialog details
// (e.g. after context compression).
//
//go:embed testdata/quizzes/curtains-quiz.txt
var QuizCurtains string
