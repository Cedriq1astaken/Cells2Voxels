const loadShader = (path) =>
  fetch(new URL(path, import.meta.url)).then((response) => {
    if (!response.ok) {
      throw new Error(`Couldn't load shader: ${path}`);
    }
    return response.text();
  });

export default loadShader;
