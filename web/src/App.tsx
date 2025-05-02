import { useQuery } from "@tanstack/react-query";
import { getSlidesSlidesGetOptions } from "./lib/api-client/@tanstack/react-query.gen";

export function App() {
  const slidesQuery = useQuery(getSlidesSlidesGetOptions());
  console.log(slidesQuery.data);
  return (
    <div>
      <h1>Slides</h1>
      <ul></ul>
    </div>
  );
}

export default App;
