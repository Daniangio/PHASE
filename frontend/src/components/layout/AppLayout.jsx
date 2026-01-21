import { Upload, Database, Server } from 'lucide-react';
import { NavLink } from 'react-router-dom';

const navItems = [
  { to: '/projects', label: 'Projects', icon: Upload },
  { to: '/results', label: 'Results', icon: Database },
  { to: '/health', label: 'Health', icon: Server },
];

export default function AppLayout({ children }) {
  return (
    <div className="flex flex-col min-h-screen bg-gray-900 text-gray-100 font-inter">
      <header className="bg-gray-800 border-b border-gray-700 shadow-lg">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-full bg-cyan-500 flex items-center justify-center text-gray-900 font-bold">
              AK
            </div>
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
          </nav>
        </div>
      </header>
      <main className="flex-grow container mx-auto px-4 py-8">{children}</main>
      <footer className="bg-gray-800 text-gray-400 text-sm text-center py-4 border-t border-gray-700">
        © {new Date().getFullYear()} PHASE
      </footer>
    </div>
  );
}
