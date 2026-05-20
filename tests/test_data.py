from frontier_lab_sprint.data import decode, encode, generate_addition_examples


def test_encode_decode_round_trip() -> None:
    text = "123 + 456 = 579\n"
    assert decode(encode(text)) == text


def test_generate_addition_examples_is_reproducible() -> None:
    first = generate_addition_examples(3, seed=7)
    second = generate_addition_examples(3, seed=7)
    assert first == second
    assert [example.text for example in first] == [
        "331 + 970 = 1301\n",
        "154 + 404 = 558\n",
        "666 + 49 = 715\n",
    ]

