import sys

from PIL import Image, ImageDraw
from PySide6 import QtCore, QtGui, QtWidgets


class DraggableRectItem(QtWidgets.QGraphicsRectItem):
    handle_size = 8.0
    handle_space = 4.0

    # Define handle positions
    handle_top_left = 1
    handle_top_middle = 2
    handle_top_right = 3
    handle_middle_left = 4
    handle_middle_right = 5
    handle_bottom_left = 6
    handle_bottom_middle = 7
    handle_bottom_right = 8

    handles = {
        handle_top_left: QtCore.QRectF(0, 0, handle_size, handle_size),
        handle_top_middle: QtCore.QRectF(0, 0, handle_size, handle_size),
        handle_top_right: QtCore.QRectF(0, 0, handle_size, handle_size),
        handle_middle_left: QtCore.QRectF(0, 0, handle_size, handle_size),
        handle_middle_right: QtCore.QRectF(0, 0, handle_size, handle_size),
        handle_bottom_left: QtCore.QRectF(0, 0, handle_size, handle_size),
        handle_bottom_middle: QtCore.QRectF(0, 0, handle_size, handle_size),
        handle_bottom_right: QtCore.QRectF(0, 0, handle_size, handle_size),
    }

    def __init__(self, *args):
        super().__init__(*args)
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsSelectable
            | QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
            # Removed ItemIsFocusable as it interferes with main window key press
        )
        self.setAcceptHoverEvents(True)
        self.current_handle = 0
        self.mouse_press_pos = QtCore.QPointF()
        self.mouse_press_rect = QtCore.QRectF()
        self._handle_pen = QtGui.QPen(QtGui.QColor("cyan"), 1, QtCore.Qt.SolidLine)
        # Change main rectangle color to red for UI feedback
        self._pen = QtGui.QPen(QtGui.QColor("red"), 2, QtCore.Qt.SolidLine)
        self.setPen(self._pen)

    def update_handles(self):
        size = self.handle_size
        bound = (
            self.rect()
        )  # Use self.rect() which is the item's local coordinates rectangle
        half_size = size / 2.0
        center_x = bound.center().x()
        center_y = bound.center().y()

        # Pass QPointF to move functions
        self.handles[self.handle_top_left].moveTopLeft(
            QtCore.QPointF(bound.left(), bound.top())
        )
        self.handles[self.handle_top_middle].moveTopLeft(
            QtCore.QPointF(center_x - half_size, bound.top())
        )
        self.handles[self.handle_top_right].moveTopLeft(
            QtCore.QPointF(bound.right() - size, bound.top())
        )
        self.handles[self.handle_middle_left].moveTopLeft(
            QtCore.QPointF(bound.left(), center_y - half_size)
        )
        self.handles[self.handle_middle_right].moveTopLeft(
            QtCore.QPointF(bound.right() - size, center_y - half_size)
        )
        self.handles[self.handle_bottom_left].moveTopLeft(
            QtCore.QPointF(bound.left(), bound.bottom() - size)
        )
        self.handles[self.handle_bottom_middle].moveTopLeft(
            QtCore.QPointF(center_x - half_size, bound.bottom() - size)
        )
        self.handles[self.handle_bottom_right].moveTopLeft(
            QtCore.QPointF(bound.right() - size, bound.bottom() - size)
        )

    def hoverMoveEvent(self, event: QtGui.QHoverEvent):
        if self.isSelected():
            for handle, rect in self.handles.items():
                if rect.contains(event.pos()):
                    self.setCursor(
                        QtCore.Qt.PointingHandCursor
                    )  # Indicate resize possibility
                    self.current_handle = handle
                    self.update()
                    return
        self.setCursor(
            QtCore.Qt.SizeAllCursor if self.isSelected() else QtCore.Qt.CrossCursor
        )
        self.current_handle = 0
        self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: QtGui.QHoverEvent):
        self.setCursor(QtCore.Qt.CrossCursor)
        self.current_handle = 0
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        self.mouse_press_pos = event.pos()
        self.mouse_press_rect = self.boundingRect()
        if self.current_handle != 0:
            # If clicking on a handle, accept the event to start resizing
            # and prevent the scene from potentially starting a new drag-draw.
            # DO NOT call super() here, as it might initiate a move.
            event.accept()
        else:
            # If clicking on the item body (not a handle), let the base class handle
            # selection and move initiation.
            super().mousePressEvent(event)
            # Accept the event so the scene doesn't try to draw a new box when moving.
            event.accept()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self.current_handle != 0:
            self.prepareGeometryChange()
            rect = self.rect()
            delta = event.pos() - self.mouse_press_pos
            orig_rect = self.mouse_press_rect

            if self.current_handle == self.handle_top_left:
                rect.setTopLeft(orig_rect.topLeft() + delta)
            elif self.current_handle == self.handle_top_right:
                rect.setTopRight(orig_rect.topRight() + delta)
            elif self.current_handle == self.handle_bottom_left:
                rect.setBottomLeft(orig_rect.bottomLeft() + delta)
            elif self.current_handle == self.handle_bottom_right:
                rect.setBottomRight(orig_rect.bottomRight() + delta)
            elif self.current_handle == self.handle_top_middle:
                rect.setTop(orig_rect.top() + delta.y())
            elif self.current_handle == self.handle_bottom_middle:
                rect.setBottom(orig_rect.bottom() + delta.y())
            elif self.current_handle == self.handle_middle_left:
                rect.setLeft(orig_rect.left() + delta.x())
            elif self.current_handle == self.handle_middle_right:
                rect.setRight(orig_rect.right() + delta.x())

            # Ensure rect is normalized (topLeft is actually top-left)
            self.setRect(rect.normalized())
            self.update_handles()
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)
            # Explicitly accept event after super() call for moving
            # to prevent it propagating to the scene.
            if event.buttons() & QtCore.Qt.LeftButton:  # Only accept if dragging
                event.accept()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self.current_handle != 0:
            self.current_handle = 0  # Reset handle state
            self.update()
            event.accept()  # Prevent deselection if clicking on handle
        super().mouseReleaseEvent(event)

    def boundingRect(self) -> QtCore.QRectF:
        o = self.handle_size + self.handle_space
        return self.rect().adjusted(-o, -o, o, o)

    def shape(self) -> QtGui.QPainterPath:
        # Make the shape include handles for hover detection
        path = QtGui.QPainterPath()
        path.addRect(self.rect())
        if self.isSelected():
            for rect in self.handles.values():
                path.addRect(rect)
        return path

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget=None,
    ):
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(self._pen)
        painter.drawRect(self.rect())

        if self.isSelected():
            painter.setPen(self._handle_pen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor("cyan")))
            self.update_handles()  # Ensure handles are in the right place before painting
            for rect in self.handles.values():
                painter.drawRect(rect)


class BoundingBoxScene(QtWidgets.QGraphicsScene):
    # Signal emitted when a new rectangle is finished being drawn
    rect_drawn = QtCore.Signal(QtCore.QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_pos = None
        self.current_temp_rect = None
        # Store reference to background image item to check against clicks
        self._background_item = None

    def set_background_item(self, item: QtWidgets.QGraphicsPixmapItem):
        """Sets the background pixmap item."""
        self._background_item = item

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            item = self.itemAt(event.scenePos(), QtGui.QTransform())
            # Start drawing ONLY if the click is on the background item or empty space
            # (item is None or item is the stored background QGraphicsPixmapItem).
            # Do NOT start drawing if clicking on an existing DraggableRectItem.
            if item is None or item == self._background_item:
                self.start_pos = event.scenePos()
                # Create a temporary rectangle for visual feedback
                self.current_temp_rect = QtWidgets.QGraphicsRectItem(
                    QtCore.QRectF(self.start_pos, self.start_pos)
                )
                self.current_temp_rect.setPen(
                    QtGui.QPen(QtGui.QColor("red"), 1, QtCore.Qt.DashLine)
                )
                self.addItem(self.current_temp_rect)
                # Important: Don't call super() here, as we are handling the press
                # to initiate drawing. The base class might try to select the background.
            else:
                # Clicked on an existing DraggableRectItem (or potentially other items).
                # Let the item handle the press for selection/moving/resizing.
                # Reset drawing state just in case.
                self.start_pos = None
                if self.current_temp_rect:
                    self.removeItem(self.current_temp_rect)
                self.current_temp_rect = None
                # Pass the event down to the item / base class
                super().mousePressEvent(event)
        else:
            # Handle other mouse buttons if necessary
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self.start_pos and self.current_temp_rect:
            current_pos = event.scenePos()
            rect = QtCore.QRectF(self.start_pos, current_pos).normalized()
            self.current_temp_rect.setRect(rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if (
            event.button() == QtCore.Qt.LeftButton
            and self.start_pos
            and self.current_temp_rect
        ):
            end_pos = event.scenePos()
            final_rect = QtCore.QRectF(self.start_pos, end_pos).normalized()
            # Remove the temporary rectangle
            self.removeItem(self.current_temp_rect)
            self.current_temp_rect = None
            self.start_pos = None

            # Emit signal only if the rectangle has non-zero width and height
            if final_rect.width() > 0 and final_rect.height() > 0:
                self.rect_drawn.emit(final_rect)
            # Even if the rect is zero size, we handled this event sequence.
            # Don't necessarily call super() unless needed for other release actions.

        else:
            # Handle release event if we weren't drawing or for other buttons
            super().mouseReleaseEvent(event)


class BoundingBoxAnnotator(
    QtWidgets.QMainWindow
):  # Changed to QMainWindow for keyPressEvent
    def __init__(self, image_path_or_pil):
        super().__init__()
        self.setWindowTitle("Bounding Box Annotator - PySide6")
        self.saved_rects = None
        self.image_item = None
        self.scene = None

        try:
            if isinstance(image_path_or_pil, Image.Image):
                img = image_path_or_pil.copy()
            else:
                img = Image.open(image_path_or_pil)

            if img.mode != "RGB":
                img = img.convert("RGB")

            # Convert PIL Image to QPixmap
            qimage = QtGui.QImage(
                img.tobytes("raw", "RGB"),
                img.width,
                img.height,
                QtGui.QImage.Format_RGB888,
            )
            pixmap = QtGui.QPixmap.fromImage(qimage)

            self.setup_ui(pixmap)

        except Exception as e:
            # Use QtWidgets.QMessageBox for error display
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load image: {e}")
            # We cannot return from __init__, so maybe set a flag or close?
            # For simplicity, let's allow the window to show but be empty/disabled
            # Or close immediately:
            QtCore.QTimer.singleShot(0, self.close)  # Close after event loop starts
            return

    def setup_ui(self, pixmap: QtGui.QPixmap):
        self.scene = BoundingBoxScene(self)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.image_item = QtWidgets.QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
        # --- Give the scene a reference to the background item ---
        self.scene.set_background_item(self.image_item)
        # ---------------------------------------------------------

        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.view.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self.view.setMouseTracking(True)  # Important for hover events
        self.view.setCursor(QtCore.Qt.CrossCursor)

        # Layout
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.addWidget(self.view)

        instructions = QtWidgets.QLabel(
            "Click and drag to draw. Click/drag box to move. Click/drag handles (cyan squares) to resize. Press DELETE to remove selected."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        button_layout = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save Annotations")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.setCentralWidget(central_widget)

        # Connect signals
        self.save_button.clicked.connect(self.save_annotations)
        self.cancel_button.clicked.connect(self.cancel)
        self.scene.rect_drawn.connect(self.add_draggable_rect)

        # Set initial size (optional, adjust as needed)
        self.resize(
            pixmap.width() + 40, pixmap.height() + 120
        )  # Add some padding for controls

        # Center window on screen
        screen_geometry = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.move(screen_geometry.center() - self.rect().center())

    def add_draggable_rect(self, rect_f: QtCore.QRectF):
        """Adds a new draggable rectangle to the scene."""
        # Deselect all other items first
        for item in self.scene.items():
            if isinstance(item, QtWidgets.QGraphicsItem):
                item.setSelected(False)

        draggable_rect = DraggableRectItem(rect_f)
        self.scene.addItem(draggable_rect)
        # Select the newly added rectangle
        draggable_rect.setSelected(True)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Handle key presses, specifically Delete/Backspace."""
        if (
            event.key() == QtCore.Qt.Key_Delete
            or event.key() == QtCore.Qt.Key_Backspace
        ):
            selected_items = self.scene.selectedItems()
            if selected_items:
                # Assuming only DraggableRectItems are selectable
                for item in selected_items:
                    if isinstance(item, DraggableRectItem):
                        self.scene.removeItem(item)
                event.accept()
                return  # Prevent further processing
        super().keyPressEvent(event)  # Pass to parent otherwise

    def save_annotations(self):
        self.saved_rects = []
        for item in self.scene.items():
            if isinstance(item, DraggableRectItem):
                rect = item.rect().normalized()  # Ensure positive width/height
                # Convert QRectF to tuple of ints
                self.saved_rects.append(
                    (
                        int(rect.left()),
                        int(rect.top()),
                        int(rect.right()),
                        int(rect.bottom()),
                    )
                )
        print(f"Saved {len(self.saved_rects)} annotations.")
        self.close()  # Close the window

    def cancel(self):
        self.saved_rects = None
        print("Annotation cancelled.")
        self.close()

    def get_annotations(self):
        # This method is called *after* the window is closed
        # The main script should check the return value of app.exec()
        # and then call this method on the window instance.
        return self.saved_rects


# --- Standalone Execution / Example Usage ---
def run_annotator(image_path_or_pil):
    """Function to run the annotator and return results."""
    # Ensure QApplication exists only once
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    annotator_window = BoundingBoxAnnotator(image_path_or_pil)
    annotator_window.show()

    # Start the event loop. exec() returns 0 if closed normally.
    app.exec()

    # Return the annotations after the window is closed
    return annotator_window.get_annotations()


if __name__ == "__main__":
    # Create a dummy black image for testing if no path provided
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
        print(f"Loading image: {image_file}")
    else:
        print("No image path provided. Creating a dummy 600x400 black image.")
        image_file = Image.new("RGB", (600, 400), color="white")
        # Draw a black border to see edges
        draw = ImageDraw.Draw(image_file)
        draw.rectangle([0, 0, 599, 399], outline="black")

    # Run the annotator UI
    results = run_annotator(image_file)

    # Process results after the UI closes
    if results is not None:
        print("\nAnnotation Results:")
        for i, rect in enumerate(results):
            print(f"  Box {i + 1}: {rect}")
    else:
        print("\nAnnotation process was cancelled or failed.")

    sys.exit(0)  # Clean exit for the application
