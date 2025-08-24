window.addEventListener('DOMContentLoaded', () => {
  if (!window.mermaid) return;
  const prefersDark = document.documentElement.getAttribute('data-md-color-scheme') === 'slate';
  mermaid.initialize({
    startOnLoad: true,
    securityLevel: 'loose',
    theme: prefersDark ? 'dark' : 'default',
    themeVariables: { fontSize: '14px' }
  });
});
