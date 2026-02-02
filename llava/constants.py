import os
import platform

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

if platform.system() == "Linux":
    PSTUTS_DIR = os.path.join(os.path.expanduser('~'), "Workspace", 'data', "Multimodal", "CVPR2020_PsTuts")

else:
    PSTUTS_DIR = os.path.join(os.path.expanduser('~'), "Workspace", 'data', "Multimodal", "CVPR2020_PsTuts")

OCR_OVERLAP_THRESHOLD = 0.7

photoshop_tools = {
    "Move": ["Move Tool", "Artboard Tool"],
    "Marquee": ["Rectangular Marquee Tool", "Elliptical Marquee Tool", 'Single Row Marquee Tool',
                'Single Column Marquee Tool'],
    "Lasso": ['Lasso Tool', 'Polygonal Lasso Tool', 'Magnetic Lasso Tool'],
    "Object Selection": ['Object Selection Tool', 'Quick Selection Tool', 'Magic Wand Tool'],
    "Cropping": ['Crop Tool', 'Perspective Crop Tool', 'Slice Tool', 'Slice Select Tool'],
    "Framing": ["Frame Tool"],
    "Eyedrop": ["Eyedropper Tool", "Color Sampler Tool", "Ruler Tool", "Note Tool", "Count Tool"],
    "Repair": ["Spot Healing Brush Tool", "Remove Tool", "Healing Brush Tool", "Patch Tool", "Content-Aware Move Tool",
               "Red Eye Tool"],

    "Pen": ["Brush Tool", "Pencil Tool", "Color Replacement Tool", "Mixer Brush Tool"],
    "Stamp": ["Clone Stamp Tool", "Pattern Stamp Tool"],
    "History Brush": ["History Brush Tool", "Art History Brush Tool"],
    "Eraser": ["Eraser Tool", "Background Eraser Tool", "Magic Eraser Tool"],
    "Paint": ["Gradient Tool", "Paint Bucket Tool"],
    "Blur": ["Blur Tool", "Sharpen Tool", "Smudge Tool"],
    "Anchor": ["Pen Tool", "Freeform Pen Tool", "Curvature Pen Tool", "Add Anchor Point Tool",
               "Delete Anchor Point Tool", "Convert Point Tool"],

    "Type": ["Horizontal Type Tool", "Vertical Type Tool", "Vertical Type Mask Tool", "Horizontal Type Mask Tool"],
    "Shapes": ["Rectangle Tool", "Ellipse Tool", "Triangle Tool", "Polygon Tool", "Line Tool", "Custom Shape Tool"],
    "Selection": ["Path Selection Tool", "Direct Selection Tool"],
    "Drag": ["Hand Tool", "Rotate Wheel Tool"],

}

photoshop_tools_flattened = [tool for tools in photoshop_tools.values() for tool in tools]


photoshop_menu_hierarchy = {
    "File": {
        "New": None,
        "Open": None,
        "Browse in Bridge": None,
        "Open as Smart Object": None,
        "Open Recent": None,
        "Close": None,
        "Close All": None,
        "Close Others": None,
        "Close and Go to Bridge": None,
        "Save": None,
        "Save As": None,
        "Save a Copy": None,
        "Revert": None,
        "Invite to Edit": None,
        "Share for Review": None,
        "Export": {
            "Quick Export as PNG": None,
            "Export As": None,
            "Export Preferences": None,
            "Save for Web": None,
            "Artboards to Files": None,
            "Artboards to PDF": None,
            "Layer Comps to Files": None,
            "Layer to Files": None,
            "Color Lookup Tables": None,
            "Data Sets as Files": None,
            "Paths to Illustrator": None,
            "Render Video": None

        },
        "Generate": {
            "Image Assets"
        },
        "Search Adobe Stock": None,
        "Search Adobe Express Templates": None,
        "Place Embedded": None,
        "Place Linked": None,
        "Package": None,
        "Automate": {
            "Batch": None,
            "PDF Presentation": None,
            "Create Droplet": None,
            "Crop and Straighten Photos": None,
            "Contact Sheet II": None,
            "Conditional Mode Change": None,
            "Fit Image": None,
            "Lens Correction": None,
            "Merge to HDR Pro": None,
            "Photomerge": None
        },
        "Scripts": {
            "Image Processor...": None,
            "Delete All Empty Layers": None,
            "Flatten All Layer Effects": None,
            "Flatten All Masks": None,
            "Script Events Manager...": None,
            "Load Files into Stack...": None,
            "Load Multiple DICOM Files...": None,
            "Statistics...": None,
            "Browse...": None
        },
        "Import": {
            "Variable Data Sets": None,
            "Video Frames to Layers": None,
            "Images from Device": None,
            "Notes": None
        },
        "Import from iPhone or iPad": {
            "Take Photo": None,
            "Scan Documents": None,
            "Add Sketches": None
        },
        "File Info...": None,
        "Version History": None,
        "Print...": None,
        "Print One Copy": None
    },
    "Edit": {
        "Undo": None,
        "Redo": None,
        "Toggle Last State": None,
        "Fade...": None,
        "Cut": None,
        "Сору": None,
        "Copy Merged": None,
        "Paste": None,
        "Paste Special": None,
        "Clear": None,
        "Search": None,
        "Check Spelling...": None,
        "Find and Replace Text...": None,
        "Fill...": None,
        "Stroke...": None,
        "Content-Aware Fill...": None,
        "Generative Fill...": None,
        "Content-Aware Scale": None,
        "Puppet Warp": None,
        "Perspective Warp": None,
        "Free Transform": None,
        "Transform": None,
        "Auto-Align Layers...": None,
        "Auto-Blend Layers...": None,
        "Sky Replacement...": None,
        "Define Brush Preset...": None,
        "Define Pattern...": None,
        "Define Custom Shape...": None,
        "Purge": ['Clipboard', 'Histories', 'All', 'Video'],
        "Adobe PDF Presets...": None,
        "Presets": ['Preset Manager...', 'Migrate Presets', 'Export/Import Presets'],
        "Remote Connections...": None,
        "Color Settings...": None,
        "Assign Profile...": None,
        "Convert to Profile...": None,
        "Keyboard Shortcuts...": None,
        "Menus...": None,
        "Toolbar...": None,
        "AutoFill": None,
        "Start Dictation": None
    },
    "Image": {
        "Mode": {"Bitmap": None,
                 "Grayscale": None,
                 "Duotone": None,
                 "Indexed Color...": None,
                 "RGB Color": None,
                 "CMYK Color": None,
                 "Lab Color": None,
                 "Multichannel": None,
                 "8 Bits/Channel": None,
                 "16 Bits/Channel": None,
                 "32 Bits/Channel": None,
                 "Color Table...": None
                 },
        "Adjustments": {
            "Brightness/Contrast...": None,
            "Levels...": None,
            "Curves...": None,
            "Exposure...": None,
            "Vibrance...": None,
            "Hue/Saturation...": None,
            "Color Balance...": None,
            "Black & White...": None,
            "Photo Filter...": None,
            "Channel Mixer...": None,
            "Color Lookup...": None,
            "Invert": None,
            "Posterize...": None,
            "Threshold...": None,
            "Gradient Map...": None,
            "Selective Color...": None,
            "Shadows/Highlights...": None,
            "HDR Toning...": None,
            "Desaturate": None,
            "Match Color...": None,
            "Replace Color...": None,
            "Equalize": None
        },
        "Auto Tone": None,
        "Auto Contrast": None,
        "Auto Color": None,
        "Image Size...": None,
        "Canvas Size...": None,
        "Image Rotation": {"180°": None,
                           "90° Clockwise": None,
                           "90° Counter Clockwise": None,
                           "Arbitrary...": None,
                           "Flip Canvas Horizontal": None,
                           "Flip Canvas Vertical": None,
                           },

        "Image Rotation": None,
        "Crop Trim...": None,
        "Reveal All": None,
        "Duplicate...": None,
        "Apply Image...": None,
        "Calculations...": None,
        "Variables": None,
        "Apply Data Set...": None,
        "Trap...": None,
        "Analysis": {
            "Set Measurement Scale": None,
            "Select Data Points": None,
            "Record Measurements": None,
            "Ruler Tool": None,
            "Count Tool": None,
            "Place Scale Marker...": None
        },

    },
    "Layer": {
        "New"
        "Copy CSS": None,
        "Copy SVG": None,
        "Duplicate Layer...": None,
        "Delete": None,
        "Quick Export as PNG": None,
        "Export As...": None,
        "Rename Layer...": None,
        "Layer Style": None,
        "Smart Filter": None,
        "New Fill Layer": None,
        "New Adjustment Layer": None,
        "Layer Content Options...": None,
        "Layer Mask": None,
        "Vector Mask": None,
        "Create Clipping Mask": None,
        "Mask All Objects": None,
        "Smart Objects": None,
        "Video Layers": None,
        "Rasterize": None,
        "New Layer Based Slice": None,
        "Group Layers": None,
        "Ungroup Layers": None,
        "Hide Layers": None,
        "Arrange": None,
        "Combine Shapes": None,
        "Align": None,
        "Distribute": None,
        "Lock Layers...": None,
        "Link Layers": None,
        "Select Linked Layers": None,
        "Merge Layers": None,
        "Merge Visible": None,
        "Flatten Image": None,
        "Matting": None,
    },
    "Type": {
        "More from Adobe Fonts...": None,
        "Panels": None,
        "Anti-Alias": None,
        "Orientation": None,
        "OpenType": None,
        "Extrude to 3D": None,
        "Create Work Path": None,
        "Convert to Shape": None,
        "Rasterize Type Layer": None,
        "Convert Text Shape Type": None,
        "Warp Text...": None,
        "Match Font...": None,
        "Font Preview Size": None,
        "Language Options": None,
        "Update All Text Layers": None,
        "Manage Missing Fonts": None,
        "Paste Lorem Ipsum": None,
        "Load Default Type Styles": None,
        "Save Default Type Styles": None,
    },
    "Select": {
        "All": None,
        "Deselect": None,
        "Reselect": None,
        "Inverse": None,
        "All Layers": None,
        "Deselect Layers": None,
        "Find Layers": None,
        "Isolate Layers": None,
        "Color Range...": None,
        "Focus Area...": None,
        "Subject": None,
        "Sky": None,
        "Select and Mask...": None,
        "Modify": None,
        "Grow": None,
        "Similar": None,
        "Transform Selection": None,
        "Edit in Quick Mask Mode": None,
        "Load Selection...": None,
        "Save Selection...": None,
        "New 3D Extrusion": None
    },
    "Filter": {
        "Last Filter": None,
        "Convert for Smart Filters": None,
        "Neural Filters...": None,
        "Filter Gallery...": None,
        "Adaptive Wide Angle...": None,
        "Camera Raw Filter...": None,
        "Lens Correction...": None,
        "Liquify...": None,
        "Vanishing Point...": None,
        "3D": None,
        "Blur": None,
        "Blur Gallery": None,
        "Distort": None,
        "Noise": None,
        "Pixelate": None,
        "Render": None,
        "Sharpen": None,
        "Stylize": None,
        "Video": None,
        "Other": None,
    },
    "View": {
        "Proof Setup": None,
        "Proof Colors": None,
        "Gamut Warning": None,
        "Pixel Aspect Ratio": None,
        "Pixel Aspect Ratio Correction": None,
        "32-bit Preview Options...": None,
        "Zoom In Zoom Out": None,
        "Fit on Screen": None,
        "Fit Layer(s) on Screen Fit Artboard on Screen": None,
        "100%": None,
        "200%": None,
        "Print Size": None,
        "Actual Size": None,
        "Flip Horizontal": None,
        "Pattern Preview": None,
        "Screen Mode": None,
        "Extras": None,
        "Show": None,
        "Rulers": None,
        "snap": None,
        "Snap To": None,
        "Guides": None,
        "Lock Slices": None,
        "Clear Slices": None,
    },
    "Plugins": {
        "Plugins Panel": None,
        "Browse Plugins...": None,
        "Manage Plugins...": None,
    },
    "Window": {
        "Arrange": None,
        "Workspace": None,
        "3D": None,
        "Actions": None,
        "Adjustments": None,
        "Brush Settings": None,
        "Brushes": None,
        "Channels": None,
        "Character": None,
        "Character Styles": None,
        "Clone Source": None,
        "Color": None,
        "Comments": None,
        "Content Credentials (Beta)": None,
        "Glyphs": None,
        "Gradients": None,
        "Histogram": None,
        "History": None,
        "Info": None,
        "Layer Comps": None,
        "Layers": None,
        "Libraries": None,
        "Materials": None,
        "Measurement Log": None,
        "Navigator": None,
        "Notes": None,
        "Paragraph": None,
        "Paragraph Styles": None,
        "Paths": None,
        "Patterns": None,
        "Properties": None,
        "Shapes": None,
        "Styles": None,
        "Swatches": None,
        "Timeline": None,
        "Tool Presets": None,
        "Version History": None,
        "Application Frame": None,
        "Options": None,
        "Tools": None,
        "Contextual Task Bar": None
    }

}

params_ui_detector = {'min-grad': 15, 'ffl-block': 5, 'min-ele-area': 200,
                      'merge-contained-ele': True, 'merge-line-to-paragraph': False, 'remove-bar': True}

photoshop_tools_string = "Tools in Photoshop:\n\n"


photoshop_tool2category = {}
for category, tool_names_li in photoshop_tools.items():
    photoshop_tools_string += f"{category}\n"
    for tool_name in tool_names_li:
        photoshop_tools_string += f"  - {tool_name}\n"
        photoshop_tool2category[tool_name] = category

questions_caption = [
    "Use 2 to 3 sentences to describe what action the user is performing. Clearly state the appearance of the "
    "object, such as its color, shape, and status. ",
    "What is the user doing in this video?",
    "Briefly describe what the user is doing in this Photoshop video.",
    "Provide a detailed description of the user's actions in around 3 sentences, including specific attributes "
    "like "
    "the color, shape, and current state of any objects involved.",
    "Can you identify the user activities within this video?",
    "In a concise manner, explain the user's activities in this Photoshop tutorial video.",
    "In two to three sentences, elaborate on the user's actions, ensuring to include details about the object's appearance, such as its color, shape, and condition.",
    "What action is the user taking in this particular video?",
    "Summarize the actions of the user in this Photoshop tutorial."
]

questions_categories = [
    "What category of tools is the user working with in this video?",
    "Which category of tools is the user using in this video?",
    "Can you identify the category of tools the user is using in this video?",
    "What category of tools does the user use in this video?",
]

questions_tools = [
    "What Photoshop tools is the user using in this video?",
    "Which Photoshop tools are being utilized by the user?",
    "What are the Photoshop tools employed by the user in this video?",
    "Can you identify the Photoshop tools the user is using in this video?",
    "What Photoshop tools does the user use in this video?",
    "Which tools in Photoshop is the user working with?",
    "What are the tools in Photoshop that the user is applying in this video?"
]

print("Initialized constants")
