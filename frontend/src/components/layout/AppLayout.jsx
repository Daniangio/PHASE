import { Upload, Database, Server, Moon, Sun, X } from 'lucide-react';
import { NavLink } from 'react-router-dom';
import { useEffect, useState } from 'react';
import SystemWorkspaceNav from './SystemWorkspaceNav';

const navItems = [
  { to: '/projects', label: 'Projects', icon: Upload },
  { to: '/results', label: 'Results', icon: Database },
  { to: '/health', label: 'Health', icon: Server },
];

export default function AppLayout({ children }) {
  const [logoOpen, setLogoOpen] = useState(false);
  const [lightMode, setLightMode] = useState(() => {
    try {
      return window.localStorage.getItem('phase-theme') === 'light';
    } catch (error) {
      return false;
    }
  });

  useEffect(() => {
    if (!logoOpen) return undefined;
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') setLogoOpen(false);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [logoOpen]);

  useEffect(() => {
    document.documentElement.classList.toggle('phase-light', lightMode);
    document.body.classList.toggle('phase-light', lightMode);
    try {
      window.localStorage.setItem('phase-theme', lightMode ? 'light' : 'dark');
    } catch (error) {
      // Theme persistence is optional (for example in restricted browser contexts).
    }
  }, [lightMode]);

  return (
    <div className="flex flex-col min-h-screen bg-gray-900 text-gray-100 font-inter overflow-x-hidden">
      {logoOpen && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-label="PHASE logo"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setLogoOpen(false);
          }}
        >
          <div className="relative w-full max-w-5xl overflow-hidden rounded-2xl border border-gray-600 bg-white shadow-2xl">
            <button
              type="button"
              onClick={() => setLogoOpen(false)}
              className="absolute right-3 top-3 z-10 rounded-full bg-gray-900/80 p-2 text-white hover:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-cyan-400"
              aria-label="Close enlarged logo"
            >
              <X className="h-5 w-5" />
            </button>
            <img src="/logo.png" alt="PHASE - Protein HAmiltonian for Sampling of Ensembles" className="block max-h-[82vh] w-full object-contain" />
          </div>
        </div>
      )}
      <header className="bg-gray-800 border-b border-gray-700 shadow-lg">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <button
              type="button"
              onClick={() => setLogoOpen(true)}
              className="h-12 w-12 shrink-0 overflow-hidden rounded-full border-2 border-cyan-400/80 bg-white shadow-md transition-transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-offset-2 focus:ring-offset-gray-800"
              aria-label="Enlarge PHASE logo"
              title="Enlarge PHASE logo"
            >
              <img src="/logo.png" alt="PHASE" className="h-full w-full object-cover object-center" />
            </button>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white">PHASE</h1>
              <p className="text-xs text-gray-400">Causal Analysis Pipeline</p>
            </div>
          </div>
          <nav className="flex items-center space-x-2">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `flex items-center space-x-2 px-3 py-2 rounded-md font-medium transition-colors ${
                    isActive
                      ? 'bg-cyan-600 text-white'
                      : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  }`
                }
              >
                <Icon className="h-4 w-4" />
                <span>{label}</span>
              </NavLink>
            ))}
            <button
              type="button"
              onClick={() => setLightMode((enabled) => !enabled)}
              className="flex items-center space-x-2 px-3 py-2 rounded-md font-medium text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
              title={lightMode ? 'Switch to dark mode' : 'Switch to light mode'}
              aria-label={lightMode ? 'Switch to dark mode' : 'Switch to light mode'}
            >
              {lightMode ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
              <span>{lightMode ? 'Dark' : 'Light'}</span>
            </button>
          </nav>
        </div>
      </header>
      <main className="flex-grow container mx-auto px-4 py-8 min-h-0">
        <SystemWorkspaceNav />
        {children}
      </main>
      <footer className="bg-gray-800 text-gray-400 text-sm text-center py-4 border-t border-gray-700">
        © {new Date().getFullYear()} PHASE
      </footer>
    </div>
  );
}
