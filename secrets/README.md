# Gmail OAuth files (private)

Save the OAuth desktop-client file downloaded from Google Cloud as
`credentials.json` in this folder:

```text
secrets/credentials.json
```

The first sign-in creates `token.json` here automatically. Both files provide
access to the connected Gmail account and are intentionally excluded from Git.
Do not share, email, or upload them.

If you change the Gmail permissions configured in Google Cloud, delete only
`token.json` and run the watcher again to approve the new permissions.
