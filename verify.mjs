import { readFileSync } from "node:fs";
import { forward } from "./src/lib/vim.js";

const weights = JSON.parse(readFileSync(new URL("./src/lib/weights.json", import.meta.url)));
const check = JSON.parse(readFileSync(new URL("./src/lib/check.json", import.meta.url)));
const examples = JSON.parse(readFileSync(new URL("./src/lib/examples.json", import.meta.url)));

const result = forward(weights, check.image);
let worst = 0;
result.logits.forEach((value, index) => {
	worst = Math.max(worst, Math.abs(value - check.logits[index]));
});
if (worst > 1e-3) {
	console.error("logit mismatch", worst, result.logits, check.logits);
	process.exit(1);
}

for (const example of examples.examples) {
	const pred = forward(weights, example.image).logits;
	const chosen = pred.indexOf(Math.max(...pred));
	if (chosen !== example.classIndex) {
		console.error("wrong class", example.label, pred);
		process.exit(1);
	}
}

const cls = weights.meta.clsAt;
const forwardRow = result.blocks[0].forward.influence[cls];
const backwardRow = result.blocks[0].backward.influence[cls];
const forwardLeak = forwardRow.slice(cls + 1).reduce((sum, value) => sum + value, 0);
const backwardLeak = backwardRow.slice(0, cls).reduce((sum, value) => sum + value, 0);
if (forwardLeak > 1e-8 || backwardLeak > 1e-8) {
	console.error("scan mask leaked", { forwardLeak, backwardLeak, forwardRow, backwardRow });
	process.exit(1);
}
console.log("logits match", worst.toExponential(2));
console.log("classes", examples.examples.map((example) => example.label).join(", "));
console.log("cls forward row", forwardRow.map((value) => value.toFixed(2)).join(" "));
console.log("cls backward row", backwardRow.map((value) => value.toFixed(2)).join(" "));
