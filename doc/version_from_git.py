import subprocess
from distutils.version import StrictVersion


def version_number_from_git(tag_prefix='release/', sha_length=10, version_format="{version}.dev{commits}+{sha}"):
    def get_released_versions():
        tags = sorted(subprocess.getoutput('git tag').split('\n'))
        versions = [t[len(tag_prefix):] for t in tags if t.startswith(tag_prefix)]
        return versions

    def tag_from_version(v):
        return tag_prefix + v

    def increment_version(v):
        parsed_version = [int(i) for i in v.split('.')]
        parsed_version[-1] += 1
        return '.'.join(str(i) for i in parsed_version)

    version_strings = get_released_versions()
    version_strings.sort(key=StrictVersion)
    latest_release = version_strings[-1]
    commits_since_tag = subprocess.getoutput('git rev-list {}..HEAD --count'.format(tag_from_version(latest_release)))
    sha = subprocess.getoutput('git rev-parse HEAD')[:sha_length]
    is_dirty = len(subprocess.getoutput("git status --untracked-files=no -s")) > 0

    if int(commits_since_tag) == 0:
        version_string = latest_release
    else:
        next_version = increment_version(latest_release)
        version_string = version_format.format(version=next_version, commits=commits_since_tag, sha=sha)

    if is_dirty:
        version_string += ".dirty"
    return version_string
