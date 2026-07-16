const fs = require('fs');
const path = require('path');
const modelDir = path.join(__dirname, '..', 'model');
const index = JSON.parse(fs.readFileSync(path.join(modelDir, 'index.json'), 'utf8'));
for (const name of index) {
  const mf = path.join(modelDir, name, 'nca_manifest.json');
  if (!fs.existsSync(mf)) { console.log(name, '— NO MANIFEST'); continue; }
  const j = JSON.parse(fs.readFileSync(mf, 'utf8'));
  const m = j.meta;
  const percShape = j.perception['perceive.weight'].shape;
  const lppnKeys = Object.keys(j.lppn);
  console.log(`${name}  ch=${m.channels}  coarse=${m.coarse_size}  target=${m.target_size}  scale=${m.scale}  perc=[${percShape}]  lppn_layers=${lppnKeys.length}  seed_r=${m.seed_radius||'N/A'}  nfreq=${m.num_frequencies||'N/A'}`);
}
