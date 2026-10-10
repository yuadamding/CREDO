"""Explicit CREDO no-runtime-deadline policy; host Python 3.6 compatible.

Network calls and launch handshakes may remain bounded. Neither elapsed time
nor queue waiting is an application failure or permission to stop the Job.
"""
POLICY = 'no_runtime_deadline_user_20260919'
FORBIDDEN_PLAN_KEYS = frozenset((
    'maximum_wall_seconds', 'maximum_job_seconds', 'runtime_deadline_seconds',
    'evaluation_deadline_seconds', 'queue_wait_deadline_seconds',
    'inherited_wall_seconds_limit', 'startup_release_deadline_seconds'))


def validate_plan(plan):
    if plan.get('runtime_deadline_policy') != POLICY:
        raise RuntimeError('Explicit no-runtime-deadline policy required')
    if set(plan) & FORBIDDEN_PLAN_KEYS:
        raise RuntimeError('Runtime deadline fields are forbidden by user policy')
    if plan.get('automatic_retries') != 0:
        raise RuntimeError('No-runtime-deadline is not automatic retry authorization')


def validate_job(job):
    if job is None:
        return  # Exact UID/existence validation remains the lifecycle's duty.
    spec = job.get('spec', {})
    if 'activeDeadlineSeconds' in spec or 'activeDeadlineSeconds' in spec.get('template', {}).get('spec', {}):
        raise RuntimeError('Runtime deadline supplied or injected into Job/Pod')


def validate_pod(pod):
    if 'activeDeadlineSeconds' in pod.get('spec', {}):
        raise RuntimeError('Runtime deadline injected into actual worker Pod')
