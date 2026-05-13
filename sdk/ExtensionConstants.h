// ExtensionConstants.h
//
// Constants for the forehead/hairline extension of the MediaPipe face mesh.
// Mirror of python/constants.py — if you change one, regenerate the other.
//
// Layout of the extended 502-vertex array consumed by the SDK:
//   [0   .. 467]  MediaPipe canonical landmarks  (3dsMax export order in OBJ)
//   [468 .. 484]  middle row (17 verts)          (identity mapping)
//   [485 .. 501]  hairline row (17 verts)        (identity mapping)
//
// Drop into the SDK in place of the existing hardcode_data.h, or include
// alongside it and use the extended map where 468 wouldn't suffice.

#ifndef HEAD3D_EXTENSION_CONSTANTS_H
#define HEAD3D_EXTENSION_CONSTANTS_H

#include <cstdint>

namespace head3d {

constexpr int kNumMP        = 468;
constexpr int kNumAnchors   = 17;
constexpr int kNumExtension = 2 * kNumAnchors;   // 34
constexpr int kNumTotal     = kNumMP + kNumExtension; // 502

constexpr int kMiddleStart   = kNumMP;                 // 468
constexpr int kHairlineStart = kNumMP + kNumAnchors;   // 485

// MediaPipe canonical landmark indices that mark the upper boundary of the
// face mesh, ordered left-to-right when viewing the face from the front.
// Each anchor i has paired vertices middle[i] = 468+i and hairline[i] = 485+i.
constexpr int kMpTopAnchors[kNumAnchors] = {
    127, 234, 162,  21,  54, 103,  67, 109,  10,
    338, 297, 332, 284, 251, 389, 356, 454,
};

// Replaces the original `indexMap[468]`. Same first 468 entries as the SDK's
// hardcode_data.h; the trailing 34 are identity (468..501) for the extension.
constexpr int kIndexMap502[kNumTotal] = {
    127,  34, 139,  11,   0,  37, 232, 231, 120,  72,
     39, 128, 121,  47, 104,  69,  67, 175, 171, 148,
    118,  50, 101,  73,  40,   9, 151, 108,  48, 115,
    131, 194, 204, 211,  74, 185,  80,  42, 183,  92,
    186, 230, 229, 202, 212, 214,  83,  18,  17,  76,
     61, 146, 160,  29,  30,  56, 157, 173, 106, 135,
    192, 203, 165,  98,  21,  71,  68,  51,  45,   4,
    144,  24,  23,  77,  91, 205, 187, 201, 200, 182,
     90, 181,  85,  84, 206,  36, 140, 193, 189, 244,
    159, 158,  28, 247, 246, 161, 236,   3, 196,  54,
    168,   8, 117, 228,  31,  55,  97,  99, 126, 100,
    166,  79, 218, 155, 154,  26, 209,  49, 136, 150,
    217, 223,  52,  53, 134, 170,  43, 119, 226, 130,
     63, 238,  20, 242,  46,  70, 156,  78,  62,  96,
    143, 227, 123, 111,  44, 125,  19, 216, 153,  22,
    167, 208, 142,  57,  60,  35, 113,  27, 210, 225,
    137, 116,  41,  38, 129,  64, 240, 102, 207, 184,
    169, 149, 176, 105,  66, 122,   6, 147,  65, 107,
     89, 180,  93,  15,  86,  14,  87, 145,  88, 179,
     95, 138, 172, 215,  58, 219,  81, 195, 199,  82,
    163, 110, 234, 109, 235, 191, 222, 141, 221, 197,
     25,   7,  33, 220, 237, 245, 162, 188, 174,   2,
    241, 164,  12,  13, 198, 133, 112, 243, 239, 190,
     32, 178, 132, 177,   1, 213,  59,  94,  75, 224,
    233, 114, 124, 356, 389, 368, 302, 267, 452, 350,
    349, 303, 269, 357, 343, 277, 453, 333, 332, 297,
    152, 377, 347, 348, 330, 304, 270, 336, 337, 278,
    279, 360, 418, 262, 431, 408, 409, 310, 415, 407,
    410, 450, 422, 430, 434, 313, 314, 306, 307, 375,
    387, 388, 260, 286, 414, 398, 335, 406, 364, 367,
    416, 423, 358, 327, 251, 284, 298, 281,   5, 373,
    374, 253, 320, 321, 425, 427, 411, 421, 405, 404,
    315,  16, 426, 266, 400, 369, 322, 391, 417, 465,
    464, 386, 257, 258, 466, 456, 399, 419, 285, 346,
    340, 261, 413, 441, 460, 328, 355, 371, 329, 392,
    439, 438, 382, 341, 256, 429, 420, 394, 379, 437,
    443, 444, 283, 275, 440, 363, 338, 273, 451, 446,
    342, 467, 293, 334, 282, 458, 461, 462, 276, 353,
    383, 308, 324, 325, 300, 372, 345, 447, 352, 274,
    248, 436, 381, 252, 393, 428, 287, 250, 384, 265,
    259, 424, 292, 366, 271, 294, 455, 272, 432, 395,
    299, 351, 280, 319, 295, 296, 403, 323, 454, 316,
    380, 318, 402, 365, 435, 397, 344, 311, 291, 396,
    268, 445, 254, 339, 449, 264,  10, 442, 370, 263,
    255, 359, 412, 301, 378, 326, 457, 362, 459, 463,
    354, 401, 361, 309, 376, 433, 289, 305, 448, 290,
    288, 249, 103, 385, 331, 317, 312, 390, 468, 469,
    470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
    480, 481, 482, 483, 484, 485, 486, 487, 488, 489,
    490, 491, 492, 493, 494, 495, 496, 497, 498, 499,
    500, 501,
};

} // namespace head3d

#endif // HEAD3D_EXTENSION_CONSTANTS_H
