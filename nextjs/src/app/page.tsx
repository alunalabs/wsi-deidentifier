"use client";
import { SlideAnnotatorNoSSR } from "@/components/SlideAnnotatorNoSSR";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  deidentifyAllSlidesSlidesDeidentifyAllPostMutation,
  getLabelStatsLabelStatsGetOptions,
  getSlidesSlidesGetOptions,
} from "@/lib/api-client/@tanstack/react-query.gen";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Toaster, toast } from "sonner";

export default function Home() {
  const queryClient = useQueryClient();
  const slidesQuery = useQuery(getSlidesSlidesGetOptions());
  const [scrolled, setScrolled] = useState(false);

  // Set up the deidentify all mutation
  const deidentifyMutation = useMutation({
    ...deidentifyAllSlidesSlidesDeidentifyAllPostMutation(),
    onSuccess: () => {
      toast.success("All slides have been deidentified successfully!");
      // Refetch slides data and label stats to update the UI
      queryClient.invalidateQueries({
        queryKey: getSlidesSlidesGetOptions().queryKey,
      });
      queryClient.invalidateQueries({
        queryKey: getLabelStatsLabelStatsGetOptions().queryKey,
      });
    },
    onError: (error) => {
      toast.error(`Error deidentifying slides: ${error.message}`);
    },
  });

  // Track scroll position for sticky header
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Fetch label statistics from the new API endpoint
  const labelStatsQuery = useQuery(getLabelStatsLabelStatsGetOptions());

  // Prepare label stats based on the API response
  const labelStats = useMemo(() => {
    // If data is loading or there's an error, use defaults
    if (labelStatsQuery.isLoading || labelStatsQuery.isError) {
      const total = slidesQuery.data?.slides?.length || 0;
      return {
        labeled: 0,
        unlabeled: total,
        total,
        allLabeled: false,
      };
    }

    // Use the data from the API (with safety checks)
    const data = labelStatsQuery.data || {
      total: 0,
      labeled: 0,
      unlabeled: 0,
      no_box_needed: 0,
    };
    const { total, labeled, unlabeled, no_box_needed } = data;

    return {
      labeled: labeled + no_box_needed, // Count both regular boxes and "no box needed" as labeled
      unlabeled,
      total,
      allLabeled: unlabeled === 0,
    };
  }, [
    labelStatsQuery.data,
    labelStatsQuery.isLoading,
    labelStatsQuery.isError,
    slidesQuery.data?.slides?.length,
  ]);

  // Handle loading and error states
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
      <Toaster />

      {/* Sticky header - position fixed when scrolled */}
      <div
        className={`sticky top-0 z-10 py-3 px-4 bg-white border-b flex justify-between items-center ${
          scrolled ? "shadow-md" : ""
        }`}
      >
        <div>
          <h1 className="text-2xl font-bold">WSI Annotator</h1>
          <p className="text-sm text-gray-500">
            {labelStats.labeled} of {labelStats.total} labeled
            {labelStats.unlabeled > 0 && (
              <span className="text-orange-500 font-medium">
                {" "}
                ({labelStats.unlabeled} unlabeled)
              </span>
            )}
          </p>
        </div>

        <Tooltip>
          <TooltipTrigger asChild>
            <span>
              <Button
                onClick={() => deidentifyMutation.mutate({})}
                disabled={
                  !labelStats.allLabeled || deidentifyMutation.isPending
                }
                variant="default"
              >
                {deidentifyMutation.isPending
                  ? "Processing..."
                  : `Deidentify (${labelStats.labeled}/${labelStats.total})`}
              </Button>
            </span>
          </TooltipTrigger>
          {!labelStats.allLabeled && (
            <TooltipContent>
              <p>
                All slides must be labeled before deidentification can be run
              </p>
            </TooltipContent>
          )}
        </Tooltip>
      </div>

      <div className="grid grid-cols-1 gap-4 mt-6">
        {slidesQuery.data.slides.map((slideStem) => (
          <SlideAnnotatorNoSSR key={slideStem} slideStem={slideStem} />
        ))}
      </div>
    </div>
  );
}
