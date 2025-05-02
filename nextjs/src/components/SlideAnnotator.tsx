"use client";

import { Button } from "@/components/ui/button";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Konva from "konva";
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Image as KonvaImage,
  Layer,
  Rect,
  Stage,
  Transformer,
} from "react-konva";
import { toast } from "sonner";
import useImage from "use-image";
import {
  getBoundingBoxBoxesSlideFilenameGetOptions,
  getBoundingBoxBoxesSlideFilenameGetQueryKey,
  getSlideImageSlidesSlideFilenameImageGetOptions,
  setBoundingBoxBoxesSlideFilenamePutMutation,
} from "../lib/api-client/@tanstack/react-query.gen";
import type { BoundingBoxInput } from "../lib/api-client/types.gen";

interface SlideAnnotatorProps {
  slideStem: string;
}

const BOX_COLOR = "#ff0000"; // Red for visibility
const STAGE_MAX_WIDTH = 500; // Max width for the canvas container
const MIN_BOX_SIZE = 5; // Minimum width/height for a box

export const SlideAnnotator: React.FC<SlideAnnotatorProps> = ({
  slideStem,
}) => {
  const queryClient = useQueryClient();

  // --- State ---
  const [imageDimensions, setImageDimensions] = useState({
    width: 0,
    height: 0,
  });
  const [scale, setScale] = useState(1);
  const [isDrawing, setIsDrawing] = useState(false);
  const [box, setBox] = useState<Konva.RectConfig | null>(null);
  const [transformerEnabled, setTransformerEnabled] = useState(false); // Start with transformer disabled
  const stageRef = useRef<Konva.Stage>(null);
  const imageLayerRef = useRef<Konva.Layer>(null);
  const boxLayerRef = useRef<Konva.Layer>(null);
  const rectRef = useRef<Konva.Rect>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  // Store initial position for drawing calculations
  const drawingStartPos = useRef<{ x: number; y: number } | null>(null);

  // --- Data Fetching ---

  // Fetch slide image data
  const imageQuery = useQuery(
    getSlideImageSlidesSlideFilenameImageGetOptions({
      path: { slide_filename: slideStem },
    })
  );
  const [konvaImage] = useImage(
    imageQuery.data?.image_data
      ? `data:image/png;base64,${imageQuery.data.image_data}`
      : "",
    "anonymous"
  );

  // Fetch existing bounding box data
  const boxQuery = useQuery(
    getBoundingBoxBoxesSlideFilenameGetOptions({
      path: { slide_filename: slideStem },
    })
  );

  // Mutation for saving/updating bounding box
  const setBoxMutation = useMutation({
    ...setBoundingBoxBoxesSlideFilenamePutMutation(),
    onSuccess: () => {
      toast.success(`Box saved for ${slideStem}.`);
      queryClient.invalidateQueries({
        queryKey: getBoundingBoxBoxesSlideFilenameGetQueryKey({
          path: { slide_filename: slideStem },
        }),
      });
      setTransformerEnabled(true); // Keep selected after save
    },
    onError: (error) => {
      console.error(`Error saving box for ${slideStem}:`, error);
      toast.error(`Failed to save box: ${error.message}`);
    },
  });

  // --- Effects ---

  // Update Konva image dimensions and scale when image loads
  useEffect(() => {
    if (konvaImage) {
      const originalWidth = konvaImage.width;
      const originalHeight = konvaImage.height;
      const stageWidth = Math.min(originalWidth, STAGE_MAX_WIDTH);
      const calculatedScale = stageWidth / originalWidth;
      setImageDimensions({
        width: stageWidth,
        height: originalHeight * calculatedScale,
      });
      setScale(calculatedScale);
    }
  }, [konvaImage]);

  // Load existing box when query resolves and scale is known
  useEffect(() => {
    // Avoid race condition: only load if component doesn't have a box yet
    if (
      boxQuery.data?.coords &&
      boxQuery.data.coords.length === 4 &&
      scale > 0 &&
      !box &&
      !isDrawing // Don't load if currently drawing
    ) {
      const [x0, y0, x1, y1] = boxQuery.data.coords;
      console.log(`Loading existing box for ${slideStem}:`, [x0, y0, x1, y1]);
      const loadedBox = {
        x: x0 * scale,
        y: y0 * scale,
        width: (x1 - x0) * scale,
        height: (y1 - y0) * scale,
        stroke: BOX_COLOR,
        strokeWidth: 2 / scale,
        draggable: true,
        id: `box-${slideStem}`,
      };
      // Only set if the box has valid dimensions
      if (loadedBox.width >= MIN_BOX_SIZE && loadedBox.height >= MIN_BOX_SIZE) {
        setBox(loadedBox);
        setTransformerEnabled(false); // Load existing box without immediate selection
      } else {
        console.warn(`Loaded box for ${slideStem} is too small, ignoring.`);
      }
    }
  }, [boxQuery.data, scale, slideStem, box, isDrawing]); // Add box and isDrawing dependency

  // Attach transformer when box exists and is selected
  useEffect(() => {
    if (
      transformerEnabled &&
      box &&
      rectRef.current &&
      transformerRef.current
    ) {
      transformerRef.current.nodes([rectRef.current]);
      transformerRef.current.getLayer()?.batchDraw();
    } else {
      transformerRef.current?.nodes([]); // Detach transformer
    }
  }, [box, transformerEnabled]);

  // --- Event Handlers ---

  const handleMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    // Ignore if clicking on the existing box or transformer
    if (e.target !== e.target.getStage()) {
      // If clicking the rectangle itself, enable the transformer
      if (rectRef.current && e.target === rectRef.current) {
        setTransformerEnabled(true);
      }
      return;
    }

    // Clicking on stage background:
    // 1. Deselect current box (if any)
    setTransformerEnabled(false);

    // 2. If no box exists, start drawing a new one
    if (!box) {
      const stage = e.target.getStage();
      const pos = stage?.getPointerPosition();
      if (!pos) return;

      setIsDrawing(true);
      drawingStartPos.current = pos; // Store start position

      setBox({
        x: pos.x,
        y: pos.y,
        width: 0,
        height: 0,
        stroke: BOX_COLOR,
        strokeWidth: 2 / scale,
        draggable: true,
        id: `box-${slideStem}`,
      });
    }
  };

  const handleMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (!isDrawing || !drawingStartPos.current) return;

    const stage = e.target.getStage();
    const pos = stage?.getPointerPosition();
    if (!pos) return;

    const startPos = drawingStartPos.current;
    // Calculate new position/dimensions relative to the starting point
    const newX = Math.min(pos.x, startPos.x);
    const newY = Math.min(pos.y, startPos.y);
    const newWidth = Math.abs(pos.x - startPos.x);
    const newHeight = Math.abs(pos.y - startPos.y);

    setBox((prevBox) => ({
      ...(prevBox as Konva.RectConfig),
      x: newX,
      y: newY,
      width: newWidth,
      height: newHeight,
    }));
  };

  const handleMouseUp = () => {
    if (isDrawing) {
      setIsDrawing(false);
      drawingStartPos.current = null;

      // Check if the drawn box is too small
      if (box && (box.width! < MIN_BOX_SIZE || box.height! < MIN_BOX_SIZE)) {
        console.log("Drawn box too small, removing.");
        setBox(null); // Remove the tiny box
        setTransformerEnabled(false);
      } else if (box) {
        // Valid box drawn, enable transformer for it
        setTransformerEnabled(true);
      }
    }
  };

  // Update box state when transformed
  const handleTransformEnd = useCallback(() => {
    const node = rectRef.current;
    if (node) {
      const scaleX = node.scaleX();
      const scaleY = node.scaleY();

      node.scaleX(1); // Reset scale after applying
      node.scaleY(1);

      const newAttrs = {
        ...box, // Spread previous attrs like stroke, id etc.
        x: node.x(),
        y: node.y(),
        width: Math.max(MIN_BOX_SIZE, node.width() * scaleX), // Enforce min size
        height: Math.max(MIN_BOX_SIZE, node.height() * scaleY), // Enforce min size
        draggable: true, // Ensure still draggable
      };
      setBox(newAttrs);
      setTransformerEnabled(true); // Keep transformer enabled after transform
    }
  }, [box]); // Depend on box state

  // Update box state when dragged
  const handleDragEnd = useCallback(
    (e: Konva.KonvaEventObject<DragEvent>) => {
      if (!box) return; // Should not happen if dragging, but safe check
      setBox({
        ...box,
        x: e.target.x(),
        y: e.target.y(),
      });
      setTransformerEnabled(true); // Keep transformer enabled after drag
    },
    [box] // Depend on box state
  );

  // Handle saving the current box
  const handleSave = useCallback(() => {
    if (!box || !rectRef.current || !scale) {
      toast.error("No valid box to save.");
      return;
    }

    // Use the current state of the box directly, as drag/transform updates it
    const finalX = box.x!;
    const finalY = box.y!;
    const finalWidth = box.width!;
    const finalHeight = box.height!;

    // Convert back to original image coordinates
    const x0 = Math.round(finalX / scale);
    const y0 = Math.round(finalY / scale);
    const x1 = Math.round((finalX + finalWidth) / scale);
    const y1 = Math.round((finalY + finalHeight) / scale);

    // Ensure x0 < x1 and y0 < y1 (should be guaranteed by drawing/transform logic)
    const coords: BoundingBoxInput["coords"] = [
      Math.min(x0, x1),
      Math.min(y0, y1),
      Math.max(x0, x1),
      Math.max(y0, y1),
    ];

    // Final validation before sending to backend
    if (
      coords.some((c) => isNaN(c) || !isFinite(c)) ||
      coords[2] - coords[0] < 1 || // Width >= 1 pixel original
      coords[3] - coords[1] < 1 // Height >= 1 pixel original
    ) {
      console.error("Invalid coordinates generated:", coords);
      toast.error("Cannot save invalid coordinates.");
      return;
    }

    console.log(`Saving box for ${slideStem}:`, coords);
    setBoxMutation.mutate({
      path: { slide_filename: slideStem },
      body: { coords },
    });
  }, [box, scale, setBoxMutation, slideStem]);

  // Handle deleting the box
  const handleDelete = useCallback(() => {
    setBox(null);
    setTransformerEnabled(false); // Disable transformer when box is removed
    // Optionally call API to delete box on server
    // For now, we just clear it locally. User needs to 'Save' an empty box if backend needs it.
    // A dedicated delete mutation could be added later.
    // Example: deleteBoxMutation.mutate({ path: { slide_filename: slideStem } });
    toast.info("Local box removed. Click 'Save Box' to persist deletion.");
  }, []);

  // --- Rendering ---

  if (imageQuery.isLoading)
    return (
      <div className="border p-2 rounded">Loading image for {slideStem}...</div>
    );
  if (imageQuery.isError)
    return (
      <div className="border p-2 rounded text-red-600">
        Error loading image: {imageQuery.error?.message || "Unknown error"}
      </div>
    );
  if (!konvaImage)
    return (
      <div className="border p-2 rounded">
        Processing image for {slideStem}...
      </div>
    );

  return (
    <div className="border p-4 rounded-lg shadow-md flex flex-col">
      <h3 className="text-lg font-semibold mb-2 truncate" title={slideStem}>
        {slideStem}
      </h3>
      {/* Updated Instructions */}
      <p className="text-sm text-gray-500 mb-2">
        Click and drag on empty space to draw a box. Click the box to
        select/transform.
      </p>
      <div
        style={{
          width: imageDimensions.width,
          height: imageDimensions.height,
          margin: "0 auto",
          cursor: !box && !isDrawing ? "crosshair" : "default", // Indicate draw possibility
        }}
        className="relative bg-gray-200"
      >
        <Stage
          width={imageDimensions.width}
          height={imageDimensions.height}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          // onClick={handleStageClick} // handleMouseDown now handles stage clicks
          // onTap={handleStageClick} // handleMouseDown now handles stage taps
          ref={stageRef}
          style={{ background: "#f0f0f0" }}
        >
          <Layer ref={imageLayerRef}>
            <KonvaImage
              image={konvaImage}
              width={imageDimensions.width}
              height={imageDimensions.height}
            />
          </Layer>
          <Layer ref={boxLayerRef}>
            {box && (
              <Rect
                ref={rectRef}
                {...box}
                name="bounding-box" // Keep name for potential future use
                // Click handled by stage mousedown now
                // Tap handled by stage mousedown now
                onDragEnd={handleDragEnd}
                onTransformEnd={handleTransformEnd}
                // Select box visually on hover (optional)
                onMouseEnter={(e) => {
                  const container = e.target.getStage()?.container();
                  if (container) container.style.cursor = "pointer";
                }}
                onMouseLeave={(e) => {
                  const container = e.target.getStage()?.container();
                  if (container)
                    container.style.cursor =
                      !box && !isDrawing ? "crosshair" : "default";
                }}
              />
            )}
            {/* Transformer is conditionally rendered based on state */}
            {transformerEnabled && box && (
              <Transformer
                ref={transformerRef}
                boundBoxFunc={(oldBox, newBox) => {
                  // Limit resize during transform operation
                  if (
                    newBox.width < MIN_BOX_SIZE ||
                    newBox.height < MIN_BOX_SIZE
                  ) {
                    return oldBox;
                  }
                  return newBox;
                }}
                // Add resize handles configuration if needed (defaults are usually fine)
                anchorStroke="dodgerblue"
                anchorFill="dodgerblue"
                anchorSize={8}
                borderStroke="dodgerblue"
                borderDash={[3, 3]}
                keepRatio={false} // Allow free transform
                rotateEnabled={false} // Disable rotation handle
              />
            )}
          </Layer>
        </Stage>
      </div>
      <div className="mt-4 flex justify-end space-x-2">
        {/* Keep Delete Button */}
        {box && (
          <Button
            variant="outline" // Changed variant for less emphasis
            size="sm"
            onClick={handleDelete}
            disabled={setBoxMutation.isPending}
          >
            Clear Box
          </Button>
        )}
        {/* Save Button saves current state (box or no box) */}
        <Button
          onClick={handleSave}
          // Allow saving even if box is null (to persist deletion)
          // disabled={!box || setBoxMutation.isPending}
          disabled={setBoxMutation.isPending}
          size="sm"
        >
          {setBoxMutation.isPending ? "Saving..." : "Save Box"}
        </Button>
      </div>
      {/* Status messages */}
      {boxQuery.isLoading && (
        <p className="text-xs text-gray-500 mt-1">Loading existing box...</p>
      )}
      {boxQuery.isSuccess &&
        !boxQuery.data?.coords?.length &&
        !box && ( // Show only if API had no box AND local state is empty
          <p className="text-xs text-gray-500 mt-1">No existing box found.</p>
        )}
      {boxQuery.isError && ( // Simplified error display
        <p className="text-xs text-red-500 mt-1">
          Error loading box:{" "}
          {boxQuery.error instanceof Error
            ? boxQuery.error.message
            : "Unknown error"}
        </p>
      )}
      {/* Mutation error handled by toast */}
    </div>
  );
};
