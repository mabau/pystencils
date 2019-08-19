import pystencils


def test_assignment_collection_dict_conversion():
    x, y = pystencils.fields('x,y: [2D]')

    collection_normal = pystencils.AssignmentCollection(
        [pystencils.Assignment(x.center(), y[1, 0] + y[0, 0])],
        []
    )
    collection_dict = pystencils.AssignmentCollection(
        {x.center(): y[1, 0] + y[0, 0]},
        {}
    )
    assert str(collection_normal) == str(collection_dict)
    assert collection_dict.main_assignments_dict == {x.center(): y[1, 0] + y[0, 0]}
    assert collection_dict.subexpressions_dict == {}

    collection_normal = pystencils.AssignmentCollection(
        [pystencils.Assignment(y[1, 0], x.center()),
         pystencils.Assignment(y[0, 0], x.center())],
        []
    )
    collection_dict = pystencils.AssignmentCollection(
        {y[1, 0]: x.center(),
         y[0, 0]: x.center()},
        {}
    )
    assert str(collection_normal) == str(collection_dict)
    assert collection_dict.main_assignments_dict == {y[1, 0]: x.center(),
                                                     y[0, 0]: x.center()}
    assert collection_dict.subexpressions_dict == {}
