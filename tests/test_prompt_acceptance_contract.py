import pytest
from agentic_pruning.e2e.prompt_parser import parse


@pytest.mark.parametrize('text,threshold',[
    ('SR>=1%',.01),('SR>=0.5%',.005),('SR>=0.90',.9),
    ('success rate at least 85%',.85),('success rate >=95%',.95),
    ('成功率不低于 0.90',.9),('成功率至少0.5%',.005),
])
def test_success_threshold_preserves_explicit_units(text,threshold):
    assert parse(text).sr_threshold==pytest.approx(threshold)


def test_common_layer_bounds_are_enforced_in_the_problem_spec():
    spec=parse('至少删2层，最多剪8层，成功率不低于85%')
    assert spec.min_k==2 and spec.max_k==8 and spec.sr_threshold==.85
    spec=parse('remove at least 2 layers; delete at most 8 layers; SR>=85%')
    assert spec.min_k==2 and spec.max_k==8


@pytest.mark.parametrize('text',['成功率下降不超过5%','success rate may drop by 5%'])
def test_relative_degradation_is_not_silently_treated_as_absolute_sr(text):
    with pytest.raises(ValueError,match='Relative success-rate'):parse(text)


def test_all_layers_does_not_request_multiple_task_suites():
    spec=parse('检查所有层，剪掉全部冗余层，成功率不低于90%')
    assert not spec.all_suites
    assert parse('评估所有任务集').all_suites
