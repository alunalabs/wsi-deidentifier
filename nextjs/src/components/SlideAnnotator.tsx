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
  getLabelStatsLabelStatsGetOptions,
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
  const [initialBoxCoords, setInitialBoxCoords] = useState<number[] | null>(
    null
  ); // Store initial coords from API
  const [isDirty, setIsDirty] = useState(false); // Track unsaved changes
  const [isMarkedNoBoxNeeded, setIsMarkedNoBoxNeeded] = useState(false); // Track if marked as explicitly no box
  const [transformerEnabled, setTransformerEnabled] = useState(false); // Start with transformer disabled
  const [isReady, setIsReady] = useState(false); // Track if Konva is ready
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
    onSuccess: (data, variables) => {
      toast.success(`Box saved for ${slideStem}.`);
      // When a save is successful, update the initialBoxCoords to match the saved state
      // This ensures we have the correct reference point for future dirty state checks
      setInitialBoxCoords(variables.body.coords);
      // Explicitly reset dirty state since we've successfully saved
      setIsDirty(false);
      // Refetch the box data for this specific slide
      queryClient.refetchQueries({
        queryKey: getBoundingBoxBoxesSlideFilenameGetQueryKey({
          path: { slide_filename: slideStem },
        }),
        exact: true, // Refetch this specific query only
      });
      // Invalidate label stats to update the header count
      queryClient.invalidateQueries({
        queryKey: getLabelStatsLabelStatsGetOptions().queryKey,
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
      setIsReady(true); // Mark as ready after dimensions and scale are set
    }
  }, [konvaImage]);

  // Load existing box OR "no box needed" state when query resolves and scale is known
  useEffect(() => {
    // Only attempt to load box when everything is ready
    if (
      isReady &&
      boxQuery.data?.coords &&
      boxQuery.data.coords.length === 4 &&
      scale > 0 &&
      !box &&
      !isDrawing // Don't load if currently drawing
      // && !isDirty // Prevent reloading if there are local unsaved changes (might need adjustment)
    ) {
      const [x0, y0, x1, y1] = boxQuery.data.coords;
      console.log(`Loading existing box for ${slideStem}:`, [x0, y0, x1, y1]);

      // Store the initial state regardless of what it is
      setInitialBoxCoords(boxQuery.data.coords);
      setIsDirty(false); // Reset dirty state on load

      // Handle "no box needed" state
      if (x0 === -1 && y0 === -1 && x1 === -1 && y1 === -1) {
        console.log(`Slide ${slideStem} marked as 'no box needed'.`);
        setBox(null);
        setIsMarkedNoBoxNeeded(true);
        setTransformerEnabled(false);
      }
      // Handle actual box coordinates
      else if (boxQuery.data.coords.length === 4) {
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
        setIsMarkedNoBoxNeeded(false);
        if (
          loadedBox.width >= MIN_BOX_SIZE &&
          loadedBox.height >= MIN_BOX_SIZE
        ) {
          setBox(loadedBox);
          setTransformerEnabled(false); // Load existing box without immediate selection
        } else {
          console.warn(`Loaded box for ${slideStem} is too small, ignoring.`);
          setBox(null); // Don't keep an invalid box
        }
      }
      // Handle unlabeled state (empty coords)
      else {
        console.log(`Slide ${slideStem} is unlabeled.`);
        setBox(null);
        setIsMarkedNoBoxNeeded(false);
        setTransformerEnabled(false);
      }
    }
    // Reset dirty flag if query is loading or scale is not ready
    else if (boxQuery.isLoading || !isReady) {
      setIsDirty(false);
    }
  }, [
    boxQuery.data,
    boxQuery.isLoading,
    scale,
    slideStem,
    box,
    isDrawing,
    isReady,
  ]);

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

  // --- Helper function to compare current box state with initial state ---
  const checkDirtyState = useCallback(
    (currentBox: Konva.RectConfig | null) => {
      if (!scale || initialBoxCoords === null) return false; // Not ready or no initial state

      if (isMarkedNoBoxNeeded) {
        // If marked as no-box, it's dirty if the initial state wasn't also no-box
        return !(
          initialBoxCoords.length === 4 &&
          initialBoxCoords.every((c) => c === -1)
        );
      }

      if (currentBox === null) {
        // If current box is null (unlabeled), it's dirty if the initial state wasn't also null/empty
        return initialBoxCoords.length > 0;
      }

      // Compare coordinates if there's a current box
      const x0 = Math.round(currentBox.x! / scale);
      const y0 = Math.round(currentBox.y! / scale);
      const x1 = Math.round((currentBox.x! + currentBox.width!) / scale);
      const y1 = Math.round((currentBox.y! + currentBox.height!) / scale);
      const currentCoords = [x0, y0, x1, y1];

      if (
        initialBoxCoords.length !== 4 ||
        initialBoxCoords.some((c) => c === -1)
      )
        return true; // Initial was empty or no-box, now we have one

      // Compare numerical arrays
      return !currentCoords.every(
        (val, index) => val === initialBoxCoords[index]
      );
    },
    [initialBoxCoords, scale, isMarkedNoBoxNeeded]
  );

  // --- Event Handlers ---

  const handleMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    // Prevent interaction if marked as no box needed
    if (isMarkedNoBoxNeeded) {
      console.log("Interaction disabled: Marked as 'no box needed'");
      return;
    }

    // Ignore if clicking on the existing box or transformer
    if (e.target !== e.target.getStage()) {
      // If clicking the rectangle itself, enable the transformer
      if (rectRef.current && e.target === rectRef.current) {
        console.log("Clicked on existing rectangle, enabling transformer");
        setTransformerEnabled(true);
      }
      return;
    }

    // Clicking on stage background:
    // 1. Deselect current box (if any)
    // Allow drawing even if a box exists (will replace it)
    setTransformerEnabled(false);

    // 2. Start drawing a new box (regardless of whether one already exists)
    const stage = e.target.getStage();
    const pos = stage?.getPointerPosition();
    console.log("Starting to draw new box at position:", pos);
    if (!pos) return;

    setIsDrawing(true);
    drawingStartPos.current = pos; // Store start position

    // Replace existing box with new one
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
    // Drawing a new box resets the "no box needed" state locally
    setIsMarkedNoBoxNeeded(false);
    console.log("Drawing state set:", {
      isDrawing: true,
      drawingStartPos: pos,
    });
  };

  const handleMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (!isDrawing || !drawingStartPos.current || isMarkedNoBoxNeeded) {
      return;
    }

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
        console.log("Drawn box too small, removing.", {
          width: box.width,
          height: box.height,
          boxStateBeforeRemove: box,
          minSize: MIN_BOX_SIZE,
        });
        setBox(null); // Remove the tiny box
        setTransformerEnabled(false);
        setIsDirty(checkDirtyState(null)); // Check if removing makes it dirty
      } else if (box) {
        // Valid box drawn, enable transformer for it
        console.log("Valid box drawn, enabling transformer", {
          finalBox: box,
        });
        setTransformerEnabled(true);
      }
    }
    // Update dirty state after potential box modification
    if (isDrawing) {
      setIsDirty(checkDirtyState(box));
    }
  };

  // --- Helper function to constrain box to image boundaries ---
  const constrainBoxToImage = useCallback(
    (box: Konva.RectConfig): Konva.RectConfig => {
      if (!box) return box;

      const x = Math.max(
        0,
        Math.min(box.x!, imageDimensions.width - box.width!)
      );
      const y = Math.max(
        0,
        Math.min(box.y!, imageDimensions.height - box.height!)
      );

      return {
        ...box,
        x,
        y,
      };
    },
    [imageDimensions]
  ); // Add imageDimensions as dependency since it's used inside

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

      // Constrain box to image boundaries
      const constrainedBox = constrainBoxToImage(newAttrs);
      setBox(constrainedBox);
      setIsDirty(checkDirtyState(constrainedBox));
      setTransformerEnabled(true); // Keep transformer enabled after transform
    }
  }, [box, constrainBoxToImage, checkDirtyState]); // Added imageDimensions dependency

  // Update box state when dragged
  const handleDragEnd = useCallback(
    (e: Konva.KonvaEventObject<DragEvent>) => {
      if (!box) return; // Should not happen if dragging, but safe check

      // Prevent updates if marked as no box needed (shouldn't be draggable anyway)
      if (isMarkedNoBoxNeeded) return;

      const draggedBox = {
        ...box,
        x: e.target.x(),
        y: e.target.y(),
      };

      // Constrain box to image boundaries
      const constrainedBox = constrainBoxToImage(draggedBox);
      setBox(constrainedBox);
      setIsDirty(checkDirtyState(constrainedBox));
      setTransformerEnabled(true); // Keep transformer enabled after drag
    },
    [box, constrainBoxToImage, checkDirtyState, isMarkedNoBoxNeeded] // Added dependencies
  );

  // Constrain box during drag (not just at drag end)
  const handleDragMove = useCallback(
    (e: Konva.KonvaEventObject<DragEvent>) => {
      // Prevent updates if marked as no box needed
      if (isMarkedNoBoxNeeded) return;

      const stage = e.target.getStage();
      // Use the current box state for width/height constraint checks
      const currentBoxState = rectRef.current?.attrs as
        | Konva.RectConfig
        | undefined;
      if (!stage || !currentBoxState) return;

      // Get current position
      const x = e.target.x();
      const y = e.target.y();

      // Constrain to image boundaries
      const newX = Math.max(
        0,
        Math.min(x, imageDimensions.width - currentBoxState.width!)
      );
      const newY = Math.max(
        0,
        Math.min(y, imageDimensions.height - currentBoxState.height!)
      );

      // Apply constraints during drag
      e.target.position({ x: newX, y: newY });
    },
    [imageDimensions, isMarkedNoBoxNeeded]
  );

  // Add Box button handler
  const handleAddBox = useCallback(() => {
    // Center the new box in the image
    const boxWidth = Math.min(100, imageDimensions.width / 3);
    const boxHeight = Math.min(100, imageDimensions.height / 3);
    const x = (imageDimensions.width - boxWidth) / 2;
    const y = (imageDimensions.height - boxHeight) / 2;

    const newBox = {
      x,
      y,
      width: boxWidth,
      height: boxHeight,
      stroke: BOX_COLOR,
      strokeWidth: 2 / scale,
      draggable: true,
      id: `box-${slideStem}`,
    };

    setBox(newBox);
    setIsDirty(checkDirtyState(newBox));
    setTransformerEnabled(true); // Enable transformer for new box
    setIsMarkedNoBoxNeeded(false); // Adding a box clears the no-box state
    toast.info("Box added. Adjust and save when ready.");
  }, [imageDimensions, scale, slideStem, checkDirtyState]);

  // Handle deleting the box
  const handleDelete = useCallback(() => {
    const wasAlreadyUnlabeled =
      initialBoxCoords === null || initialBoxCoords.length === 0;
    setIsMarkedNoBoxNeeded(false); // Deleting means it's now unlabeled
    setBox(null);
    setTransformerEnabled(false); // Disable transformer when box is removed

    const dirty = !wasAlreadyUnlabeled;
    setIsDirty(dirty);

    // Only call mutation if the state actually changed
    if (dirty) {
      const newCoords: [number, number, number, number] = [0, 0, 0, 0]; // Special value for deletion -> unlabeled
      setBoxMutation.mutate({
        path: { slide_filename: slideStem },
        body: { coords: newCoords },
      });
      // We don't need to update isDirty here since the onSuccess handler will do that
      toast.info("Box deleted. Saving...");
    } else {
      toast.info("Already unlabeled.");
    }
  }, [slideStem, setBoxMutation, initialBoxCoords]);

  // Handle marking as "No Box Needed"
  const handleMarkNoBoxNeeded = useCallback(() => {
    const wasAlreadyMarked = initialBoxCoords?.every((c) => c === -1);
    setIsMarkedNoBoxNeeded(true);
    setBox(null); // Visually remove box
    setTransformerEnabled(false);

    const dirty = !wasAlreadyMarked;
    setIsDirty(dirty);

    // Only call mutation if the state actually changed
    if (dirty) {
      const newCoords: [number, number, number, number] = [-1, -1, -1, -1]; // Special value for no box needed
      setBoxMutation.mutate({
        path: { slide_filename: slideStem },
        body: { coords: newCoords },
      });
      // We don't need to update isDirty here since the onSuccess handler will do that
      toast.info("Marked as 'No Box Needed'. Saving...");
    } else {
      toast.info("Already marked as 'No Box Needed'.");
    }
  }, [slideStem, setBoxMutation, initialBoxCoords]);

  // Handle saving the current box
  const handleSave = useCallback(() => {
    if (isMarkedNoBoxNeeded || !box) {
      toast.error(
        "Cannot save, no box present. Use 'Delete Box' or 'No Box Needed'."
      );
      return;
    }
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
  }, [box, scale, setBoxMutation, slideStem, isMarkedNoBoxNeeded]);

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
        {slideStem} {isDirty && <span className="text-orange-500 ml-1">*</span>}
      </h3>
      {/* Status Indicator */}
      <div className="text-sm text-gray-500 mb-2">
        Status:{" "}
        {boxQuery.isLoading ? (
          <span className="italic">Loading...</span>
        ) : isMarkedNoBoxNeeded ? (
          <span className="text-blue-600 font-medium">
            Marked &apos;No Box Needed&apos;{" "}
            {isDirty ? (
              <span className="text-orange-500 font-normal italic">
                (unsaved)
              </span>
            ) : (
              <span className="text-green-600 font-normal italic">(saved)</span>
            )}
          </span>
        ) : box ? (
          <span className="text-green-600 font-medium">
            Box Present{" "}
            {isDirty ? (
              <span className="text-orange-500 font-normal italic">
                (unsaved)
              </span>
            ) : (
              <span className="text-green-600 font-normal italic">(saved)</span>
            )}
          </span>
        ) : (
          <span className="text-gray-600 font-medium">
            Unlabeled{" "}
            {isDirty ? (
              <span className="text-orange-500 font-normal italic">
                (unsaved)
              </span>
            ) : (
              <span className="text-green-600 font-normal italic">(saved)</span>
            )}
          </span>
        )}
      </div>
      {/* Updated Instructions */}
      <p className="text-sm text-gray-500 mb-2">
        {!isMarkedNoBoxNeeded
          ? "Click/drag on empty space to draw. Click box to select/transform."
          : "This slide is marked as not needing a box."}
      </p>
      <div
        style={{
          width: imageDimensions.width,
          height: imageDimensions.height,
          margin: "0 auto",
          cursor: isMarkedNoBoxNeeded
            ? "not-allowed"
            : !box && !isDrawing
            ? "crosshair"
            : "default", // Indicate draw possibility or restriction
        }}
        className="relative bg-gray-200"
      >
        <Stage
          width={imageDimensions.width}
          height={imageDimensions.height}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          ref={stageRef}
          style={{ background: "#f0f0f0" }}
        >
          <Layer ref={imageLayerRef}>
            <KonvaImage
              image={konvaImage}
              width={imageDimensions.width}
              height={imageDimensions.height}
              listening={false}
            />
          </Layer>
          <Layer ref={boxLayerRef}>
            {box && (
              <Rect
                ref={rectRef}
                {...box}
                strokeScaleEnabled={false}
                name="bounding-box" // Keep name for potential future use
                // Only allow interactions if not marked as no box needed
                draggable={!isMarkedNoBoxNeeded}
                onDragEnd={handleDragEnd}
                onDragMove={handleDragMove}
                onTransformEnd={handleTransformEnd}
                // Select box visually on hover (optional)
                onMouseEnter={(e) => {
                  if (isMarkedNoBoxNeeded) return;
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

                  // Keep the transformer within the image bounds
                  if (newBox.x < 0) {
                    newBox.x = 0;
                  }
                  if (newBox.y < 0) {
                    newBox.y = 0;
                  }
                  if (newBox.x + newBox.width > imageDimensions.width) {
                    newBox.width = imageDimensions.width - newBox.x;
                  }
                  if (newBox.y + newBox.height > imageDimensions.height) {
                    newBox.height = imageDimensions.height - newBox.y;
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
        {/* Add Box Button */}
        {!box && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleAddBox}
            disabled={setBoxMutation.isPending}
          >
            Add Box
          </Button>
        )}
        {/* Delete Box Button */}
        {box && (
          <Button
            variant="destructive" // Changed to destructive for clarity
            size="sm"
            onClick={handleDelete}
            disabled={setBoxMutation.isPending}
          >
            Delete Box
          </Button>
        )}
        {/* Save Button saves current state (box or no box) */}
        {box && (
          <Button
            onClick={handleSave}
            disabled={setBoxMutation.isPending}
            size="sm"
          >
            {setBoxMutation.isPending ? "Saving..." : "Save Box"}
          </Button>
        )}
        {/* Mark No Box Needed Button */}
        {!box && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleMarkNoBoxNeeded}
            disabled={setBoxMutation.isPending || isMarkedNoBoxNeeded}
          >
            No Box Needed
          </Button>
        )}
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
