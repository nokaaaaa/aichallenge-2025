#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${repo_dir}/aichallenge/workspace/src"
archive="${repo_dir}/submit/aichallenge_submit.tar.gz"
launch_file="aichallenge_submit/aichallenge_submit_launch/launch/aichallenge_submit.launch.xml"
staging_dir="$(mktemp -d)"
trap 'rm -rf "${staging_dir}"' EXIT

mkdir -p "${repo_dir}/submit"

cp -a "${source_dir}/aichallenge_submit" "${staging_dir}/"

# The evaluation image does not pass control_method to the system launch, so
# its upstream default would otherwise select pure pursuit. Keep the working
# tree configurable for dev2/dev3/dev4, and force P1 only in the archive.
sed -i \
  's#    <arg name="control_method" value="$(var control_method)"/>#    <arg name="control_method" value="pp_mpc_avoidance"/>#' \
  "${staging_dir}/${launch_file}"

# Do not ship interpreter caches from a local build. They can be stale and are
# not part of the submitted source package.
tar czf "${archive}" \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*/.pytest_cache' \
  -C "${staging_dir}" \
  aichallenge_submit

# Fail early if the archive would start a different controller in evaluation.
if ! tar xOf "${archive}" "${launch_file}" | grep -Fq '<arg name="control_method" value="pp_mpc_avoidance"/>'; then
  echo "error: submission archive does not force the P1 controller" >&2
  exit 1
fi

echo "created ${archive} (P1: pp_mpc_avoidance)"
