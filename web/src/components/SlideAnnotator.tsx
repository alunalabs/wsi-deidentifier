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
import useImage from "use-image";
import {
  getBoundingBoxBoxesSlideFilenameGetOptions,
  getBoundingBoxBoxesSlideFilenameGetQueryKey,
  getSlideImageSlidesSlideFilenameImageGetOptions,
  setBoundingBoxBoxesSlideFilenamePutMutation,
} from "../lib/api-client/@tanstack/react-query.gen";
import type { BoundingBoxInput } from "../lib/api-client/types.gen";
import { Button } from "./ui/button"; // Assuming shadcn/ui Button

interface SlideAnnotatorProps {
  slideStem: string;
}

const BOX_COLOR = "#ff0000"; // Red for visibility
const STAGE_MAX_WIDTH = 500; // Max width for the canvas container

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
  const [transformerEnabled, setTransformerEnabled] = useState(true); // Enable transformer initially
  const stageRef = useRef<Konva.Stage>(null);
  const imageLayerRef = useRef<Konva.Layer>(null);
  const boxLayerRef = useRef<Konva.Layer>(null);
  const rectRef = useRef<Konva.Rect>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const startPosRef = useRef<{ x: number; y: number } | null>(null); // Add ref for start position

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
      : "", // Provide empty string if no data yet
    "anonymous" // CORS policy needs to be lowercase
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
    onSuccess: (data) => {
      console.log(`Box saved successfully for ${slideStem}:`, data);
      // Invalidate queries to refetch data after mutation
      queryClient.invalidateQueries({
        queryKey: getBoundingBoxBoxesSlideFilenameGetQueryKey({
          path: { slide_filename: slideStem },
        }),
      });
      // Optionally show feedback to user
    },
    onError: (error) => {
      console.error(`Error saving box for ${slideStem}:`, error);
      // Optionally show error feedback to user
    },
  });

  // --- Effects ---

  // Effect to update Konva image dimensions and scale when image loads
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

  // Effect to load existing box when query resolves (and scale is known)
  useEffect(() => {
    // Check if data exists, coords is an array with items, and scale is ready
    if (
      boxQuery.data?.coords &&
      boxQuery.data.coords.length === 4 &&
      scale > 0 &&
      !box
    ) {
      const [x0, y0, x1, y1] = boxQuery.data.coords;
      console.log(`Loading existing box for ${slideStem}:`, [x0, y0, x1, y1]);
      setBox({
        x: x0 * scale,
        y: y0 * scale,
        width: (x1 - x0) * scale,
        height: (y1 - y0) * scale,
        stroke: BOX_COLOR,
        strokeWidth: 2 / scale, // Adjust stroke width based on scale
        draggable: true,
        id: `box-${slideStem}`,
      });
      setTransformerEnabled(true); // Enable transformer for existing box
    }
  }, [boxQuery.data, scale, slideStem, box]);

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
      transformerRef.current?.nodes([]);
    }
  }, [box, transformerEnabled]);

  // --- Event Handlers ---

  const handleMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    // Only start drawing if no box exists yet
    if (box) return;
    // Prevent starting draw if clicking on existing shapes (though we only have one)
    if (e.target !== e.target.getStage()) return;

    setIsDrawing(true);
    setTransformerEnabled(false); // Disable transformer during draw
    const stage = e.target.getStage(); // Get stage reference
    const pos = stage?.getPointerPosition();
    if (!pos || !stage) return; // Ensure pos and stage are valid

    startPosRef.current = pos; // Store starting position in ref

    setBox({
      x: pos.x,
      y: pos.y,
      width: 0,
      height: 0,
      stroke: BOX_COLOR,
      strokeWidth: 2 / scale, // Use current scale
      draggable: true, // Make it draggable immediately
      id: `box-${slideStem}`,
    });
  };

  const handleMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    // Use the ref for start position and check isDrawing state
    if (!isDrawing || !startPosRef.current) return;

    const stage = e.target.getStage();
    const pos = stage?.getPointerPosition();
    if (!pos || !stage) return;

    const startPos = startPosRef.current;
    const newWidth = pos.x - startPos.x;
    const newHeight = pos.y - startPos.y;

    // Update the box state using functional form and startPos from ref
    setBox((prevBox) => ({
      ...(prevBox as Konva.RectConfig), // Assert prevBox is not null
      // Keep original starting point from the ref
      x: startPos.x,
      y: startPos.y,
      // Update width/height based on current mouse pos and stored start pos
      width: newWidth,
      height: newHeight,
    }));
  };

  const handleMouseUp = () => {
    if (isDrawing) {
      setIsDrawing(false);
      startPosRef.current = null; // Clear the start position ref

      // Minimal size check - prevent tiny boxes if desired
      // Check if box exists before accessing its properties
      if (box && (Math.abs(box.width!) < 5 || Math.abs(box.height!) < 5)) {
        console.log("Box too small, removing.");
        setBox(null); // Remove small box
      } else if (box) {
        // Only enable transformer if a valid box remains
        setTransformerEnabled(true); // Re-enable transformer after drawing
      }
    }
  };

  // Update box state when transformed
  const handleTransformEnd = useCallback(() => {
    const node = rectRef.current;
    if (node) {
      const scaleX = node.scaleX();
      const scaleY = node.scaleY();

      // Reset scale to avoid compounding transforms
      node.scaleX(1);
      node.scaleY(1);

      const newAttrs = {
        ...box,
        x: node.x(),
        y: node.y(),
        // Adjust width/height by scale
        width: Math.max(5, node.width() * scaleX),
        height: Math.max(5, node.height() * scaleY),
      };
      setBox(newAttrs);
      setTransformerEnabled(true); // Keep transformer enabled
    }
  }, [box]);

  // Update box state when dragged
  const handleDragEnd = useCallback(
    (e: Konva.KonvaEventObject<DragEvent>) => {
      setBox({
        ...box!,
        x: e.target.x(),
        y: e.target.y(),
      });
      setTransformerEnabled(true); // Keep transformer enabled
    },
    [box]
  );

  // Handle saving the current box
  const handleSave = () => {
    if (!box || !rectRef.current) {
      console.warn("No box to save.");
      return; // Or potentially delete existing if box is null?
    }

    // Get final coordinates from the Konva node
    const finalX = rectRef.current.x();
    const finalY = rectRef.current.y();
    const finalWidth = rectRef.current.width() * rectRef.current.scaleX(); // Apply scale
    const finalHeight = rectRef.current.height() * rectRef.current.scaleY(); // Apply scale

    // Convert back to original image coordinates
    const x0 = Math.round(finalX / scale);
    const y0 = Math.round(finalY / scale);
    const x1 = Math.round((finalX + finalWidth) / scale);
    const y1 = Math.round((finalY + finalHeight) / scale);

    // Ensure x0 < x1 and y0 < y1
    const coords: BoundingBoxInput["coords"] = [
      Math.min(x0, x1),
      Math.min(y0, y1),
      Math.max(x0, x1),
      Math.max(y0, y1),
    ];

    // Validate coordinates (simple check)
    if (
      coords.some((c) => isNaN(c) || !isFinite(c)) ||
      coords[0] >= coords[2] ||
      coords[1] >= coords[3]
    ) {
      console.error("Invalid coordinates generated:", coords);
      alert("Cannot save invalid coordinates.");
      return;
    }

    console.log(`Saving box for ${slideStem}:`, coords);
    setBoxMutation.mutate({
      path: { slide_filename: slideStem },
      body: { coords },
    });
  };

  // Handle deleting the box
  const handleDelete = () => {
    setBox(null);
    setTransformerEnabled(false);
    // TODO: Optionally call API to delete box on server if needed
    console.log(`Box deleted locally for ${slideStem}`);
    // If you want to persist deletion, call mutation with empty/null coords or a dedicated DELETE endpoint
    // setBoxMutation.mutate({ path: { slide_filename: slideStem }, body: { coords: [] } }); // Example
  };

  // Handle clicking outside the box to deselect
  const handleStageClick = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      // if click is on empty area - remove all transformers
      if (e.target === e.target.getStage()) {
        setTransformerEnabled(false);
        transformerRef.current?.nodes([]);
        return;
      }

      // do nothing if clicked NOT on our rectangle
      if (!e.target.hasName("bounding-box")) {
        return;
      }

      // clicked on transformer - do nothing
      const clickedOnTransformer =
        e.target.getParent()?.className === "Transformer";
      if (clickedOnTransformer) {
        return;
      }

      // find clicked rect by its name
      if (rectRef.current && e.target === rectRef.current) {
        setTransformerEnabled(true);
        transformerRef.current?.nodes([rectRef.current]);
      } else {
        setTransformerEnabled(false);
        transformerRef.current?.nodes([]);
      }
    },
    []
  );

  // --- Rendering ---

  if (imageQuery.isLoading)
    return (
      <div className="border p-2 rounded">Loading image for {slideStem}...</div>
    );
  if (imageQuery.isError)
    return (
      <div className="border p-2 rounded text-red-600">
        Error loading image: {imageQuery.error.message}
      </div>
    );
  if (!konvaImage)
    return (
      <div className="border p-2 rounded">
        Processing image for {slideStem}...
      </div>
    ); // Waiting for useImage hook

  return (
    <div className="border p-4 rounded-lg shadow-md flex flex-col">
      <h3 className="text-lg font-semibold mb-2 truncate" title={slideStem}>
        {slideStem}
      </h3>
      <p className="text-sm text-gray-500 mb-2">
        Hold Shift and drag to draw a box. Click box to select/transform.
      </p>
      <div
        style={{
          width: imageDimensions.width,
          height: imageDimensions.height,
          margin: "0 auto",
        }}
        className="relative bg-gray-200"
      >
        <Stage
          width={imageDimensions.width}
          height={imageDimensions.height}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onClick={handleStageClick}
          onTap={handleStageClick} // For touch devices
          ref={stageRef}
          style={{ background: "#f0f0f0" }} // Add background for clarity
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
                name="bounding-box" // Add name for selection logic
                onClick={() => setTransformerEnabled(true)} // Select on click
                onTap={() => setTransformerEnabled(true)} // Select on tap
                onDragEnd={handleDragEnd}
                onTransformEnd={handleTransformEnd}
              />
            )}
            {box && transformerEnabled && (
              <Transformer
                ref={transformerRef}
                boundBoxFunc={(oldBox, newBox) => {
                  // limit resize
                  if (newBox.width < 5 || newBox.height < 5) {
                    return oldBox;
                  }
                  return newBox;
                }}
              />
            )}
          </Layer>
        </Stage>
      </div>
      <div className="mt-4 flex justify-end space-x-2">
        {box && (
          <Button
            variant="destructive"
            size="sm"
            onClick={handleDelete}
            disabled={setBoxMutation.isPending}
          >
            Delete Box
          </Button>
        )}
        <Button
          onClick={handleSave}
          disabled={!box || setBoxMutation.isPending}
          size="sm"
        >
          {setBoxMutation.isPending ? "Saving..." : "Save Box"}
        </Button>
      </div>
      {boxQuery.isLoading && (
        <p className="text-xs text-gray-500 mt-1">Loading existing box...</p>
      )}
      {/* Check if query succeeded and coords array is empty */}
      {boxQuery.isSuccess && boxQuery.data?.coords?.length === 0 && (
        <p className="text-xs text-gray-500 mt-1">No existing box found.</p>
      )}
      {/* Check for actual errors (e.g., network, 500, or 404 for slide itself) */}
      {boxQuery.isError && boxQuery.error instanceof Error && (
        <p className="text-xs text-red-500 mt-1">
          Error loading box: {boxQuery.error.message}
        </p>
      )}
      {/* Fallback for non-Error objects */}
      {boxQuery.isError && !(boxQuery.error instanceof Error) && (
        <p className="text-xs text-red-500 mt-1">
          Error loading box: Unknown error
        </p>
      )}
      {setBoxMutation.isError && (
        <p className="text-xs text-red-500 mt-1">
          Error saving:{" "}
          {setBoxMutation.error instanceof Error
            ? setBoxMutation.error.message
            : "Unknown error"}
        </p>
      )}
    </div>
  );
};
