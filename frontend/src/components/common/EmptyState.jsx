import { FolderDown } from 'lucide-react';

export default function EmptyState({ title, description, action }) {
  return (
    <div className="border border-dashed border-gray-700 rounded-lg p-8 text-center text-gray-400">
      <FolderDown className="h-10 w-10 mx-auto mb-4 text-gray-500" />
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      <p className="text-sm mb-4">{description}</p>
      {action}
    </div>
  );
}
