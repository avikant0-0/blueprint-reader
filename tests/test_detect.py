from blueprint_reader.detect import Detection, nms, tile_origins


def test_small_image_is_one_tile():
    assert tile_origins(800, 1024, 0.2) == [0]


def test_tiles_cover_the_whole_axis_with_overlap():
    size, tile = 5400, 1024
    starts = tile_origins(size, tile, 0.2)
    assert starts[0] == 0 and starts[-1] == size - tile
    for a, b in zip(starts, starts[1:]):
        assert b - a <= tile * 0.8 + 1  # consecutive tiles overlap by >= 20%


def test_nms_merges_duplicates_from_overlapping_tiles():
    dets = [Detection("door", 0.9, 100, 100, 150, 150),
            Detection("door", 0.6, 102, 101, 151, 149),   # same door seen in the next tile
            Detection("door", 0.8, 400, 400, 450, 450),
            Detection("window", 0.7, 100, 100, 150, 150)]  # other class is kept
    kept = nms(dets, 0.5)
    assert [(d.cls, d.conf) for d in kept] == [("door", 0.9), ("door", 0.8), ("window", 0.7)]


def test_nms_keeps_touching_but_distinct_boxes():
    dets = [Detection("door", 0.9, 0, 0, 50, 50), Detection("door", 0.8, 40, 0, 90, 50)]
    assert len(nms(dets, 0.5)) == 2
