// Package testassets embeds the files from testdata/ so the server can use
// them at runtime: user profiles for the "сообщение" blob menu. The embedding
// package lives at the module root because the go:embed directive cannot
// reference files above or beside its own directory, and packages inside
// testdata/ are ignored by the go tool.
package testassets

import _ "embed"

// UserProfiles is the prototype file describing the user personas offered as
// "сообщение" blobs in the UI. Each profile is a "Профиль:" / "Текст:" pair;
// pressing a blob sends the self-introduction line as the next user message.
//
//go:embed testdata/profiles/user-profiles.txt
var UserProfiles string
