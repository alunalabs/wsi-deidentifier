import { useQuery } from "@tanstack/react-query";
import { SlideAnnotator } from "./components/SlideAnnotator";
import { getSlidesSlidesGetOptions } from "./lib/api-client/@tanstack/react-query.gen";

export function App() {
  const slidesQuery = useQuery(getSlidesSlidesGetOptions());

  if (slidesQuery.isLoading) {
    return <div>Loading slides...</div>;
  }

  if (slidesQuery.isError) {
    return <div>Error loading slides: {slidesQuery.error.message}</div>;
  }

  if (!slidesQuery.data?.slides || slidesQuery.data.slides.length === 0) {
    return <div>No slides found. Check server SLIDE_PATTERN.</div>;
  }

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">WSI Annotator</h1>
      <div className="grid grid-cols-1 gap-4">
        {slidesQuery.data.slides.map((slideStem) => (
          <SlideAnnotator key={slideStem} slideStem={slideStem} />
        ))}
      </div>
    </div>
  );
}

export default App;
