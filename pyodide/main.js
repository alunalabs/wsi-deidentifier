async function log(msg) {
  document.getElementById('log').textContent += msg + '\n';
}

async function setupPyodide() {
  const pyodide = await loadPyodide();
  await pyodide.loadPackage(['micropip', 'pillow']);
  // load project modules from repository root
  for (const mod of ['tiffparser.py', 'replace_macro.py', 'deidentify.py']) {
    const resp = await fetch('../' + mod);
    const text = await resp.text();
    await pyodide.runPython(text);
  }
  await pyodide.runPython('import sys; sys.path.append("/")');
  return pyodide;
}

async function deidentifyFile(pyodide, file, salt) {
  const data = new Uint8Array(await file.arrayBuffer());
  const inputPath = '/' + file.name;
  pyodide.FS.writeFile(inputPath, data);
  pyodide.FS.mkdir('/out', { recursive: true });
  const args = ['deidentify.py', inputPath, '-o', '/out', '-m', '/hash_mapping.csv'];
  if (salt) {
    args.push('--salt', salt);
  }
  const command = `import sys, deidentify\nsys.argv = ${JSON.stringify(args)}\ndeidentify.main()`;
  await pyodide.runPythonAsync(command);
  const outFiles = pyodide.FS.readdir('/out').filter(n => n !== '.' && n !== '..');
  const downloads = document.getElementById('downloads');
  downloads.innerHTML = '';
  for (const name of outFiles) {
    const content = pyodide.FS.readFile('/out/' + name);
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([content]));
    link.download = name;
    link.textContent = 'Download ' + name;
    downloads.appendChild(link);
    downloads.appendChild(document.createElement('br'));
  }
  const csv = pyodide.FS.readFile('/hash_mapping.csv', { encoding: 'utf8' });
  const csvLink = document.createElement('a');
  csvLink.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  csvLink.download = 'hash_mapping.csv';
  csvLink.textContent = 'Download mapping CSV';
  downloads.appendChild(csvLink);
}

(async () => {
  const pyodide = await setupPyodide();
  document.getElementById('runBtn').onclick = async () => {
    const fileInput = document.getElementById('slideFile');
    if (!fileInput.files.length) {
      alert('Select a slide file');
      return;
    }
    const salt = document.getElementById('saltInput').value;
    document.getElementById('downloads').innerHTML = '';
    await deidentifyFile(pyodide, fileInput.files[0], salt);
  };
})();
