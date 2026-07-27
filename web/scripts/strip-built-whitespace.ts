const path = new URL("../dist/assets/app.js", import.meta.url)
const source = await Bun.file(path).text()
await Bun.write(path, source.replace(/[ \t]+$/gm, ""))
