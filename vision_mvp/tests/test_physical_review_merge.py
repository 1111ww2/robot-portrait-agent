import copy
import pytest

from leica_tracker.physical_review_merge import merge_preview
from leica_tracker.physical_runtime import reviewed_profile
from test_physical_runtime import data


def draft_and_report():
    draft=data();report=copy.deepcopy(draft['intro_preview'])
    draft['status']='DRAFT';draft['intro_preview']={'status':'NOT_ACCEPTED'}
    draft['installation']['review']={k:False for k in draft['installation']['review']}
    draft['intro']['review']={k:False for k in draft['intro']['review']}
    return draft,report


def test_merge_preserves_draft_and_does_not_approve_paths_or_dynamics():
    draft,report=draft_and_report();merged=merge_preview(draft,report)
    assert merged['intro_preview']==report and merged['status']=='DRAFT'
    assert not any(merged['installation']['review'].values())
    assert not any(merged['intro']['review'].values())
    assert draft['intro_preview']['status']=='NOT_ACCEPTED'
    with pytest.raises(ValueError):reviewed_profile(merged)


@pytest.mark.parametrize('changed',['ip','target','tcp','observation'])
def test_mismatched_report_is_not_merged(changed):
    draft,report=draft_and_report()
    if changed=='ip':report['robot_ip']='192.0.2.2';report['reviewed']['robot_ip']=report['robot_ip']
    if changed=='target':draft['intro']['waypoints'][0][0]+=.1
    if changed=='tcp':draft['intro']['tcp'][2]+=.1
    if changed=='observation':report['observations'][0]['joints'][0]+=.1;report['reviewed']['observations']=copy.deepcopy(report['observations'])
    with pytest.raises(ValueError):merge_preview(draft,report)
