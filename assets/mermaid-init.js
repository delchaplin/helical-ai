window.addEventListener('DOMContentLoaded', () => {
  if (!window.mermaid) return;
  mermaid.initialize({ startOnLoad: true, securityLevel: 'loose' });
});
