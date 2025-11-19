export default function ErrorMessage({ title = 'Something went wrong', message }) {
  if (!message) return null;
  return (
    <div className="bg-red-500/10 border border-red-500/40 text-red-200 px-4 py-3 rounded-lg text-sm mb-4">
      <p className="font-semibold">{title}</p>
      <p>{message}</p>
    </div>
  );
}
