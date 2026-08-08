import pandas as pd
from alp_discrimination.workflows.conditional_feature_production import bracket, refinement_grid, full_grid, selection_grid

def test_bracket():
    c=pd.DataFrame({"number_of_events":[2,3,4,5],"worst_case_accuracy":[.7,.87,.92,.95]})
    assert bracket(c)==(4,3,4)

def test_no_bracket():
    c=pd.DataFrame({"number_of_events":[20,30],"worst_case_accuracy":[.7,.8]})
    assert bracket(c)==(None,30,None)

def test_refinement_unit_grid():
    g=refinement_grid(120,130)
    assert all(x in g for x in range(120,131))
    assert max(g)>130

def test_full_grid():
    g=full_grid(142)
    assert all(x in g for x in range(134,158))
    assert max(g)>=213

def test_low_full_grid():
    g=full_grid(4)
    assert min(g)==2 and 4 in g

def test_selection_grid():
    assert selection_grid(4,full_grid(4))==(3,4,5)
    assert selection_grid(2,full_grid(2))==(2,3)
