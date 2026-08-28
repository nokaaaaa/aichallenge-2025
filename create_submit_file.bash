#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${repo_dir}/aichallenge/workspace/src"
archive="${repo_dir}/submit/aichallenge_submit.tar.gz"
launch_file="aichallenge_submit/aichallenge_submit_launch/launch/aichallenge_submit.launch.xml"
pp_mpc_launch_file="aichallenge_submit/aichallenge_submit_launch/launch/control/pp_mpc_avoidance.launch.xml"
staging_dir="$(mktemp -d)"
control_method="pp_mpc_avoidance"
trap 'rm -rf "${staging_dir}"' EXIT

usage() {
  cat <<'EOF'
Usage: ./create_submit_file.bash

Build submit/aichallenge_submit.tar.gz.
The archive always uses the P1 behavior: pp_mpc_avoidance with the ego-vehicle gate disabled.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "${repo_dir}/submit"

cp -a "${source_dir}/aichallenge_submit" "${staging_dir}/"

# The evaluation image does not pass control_method to the system launch.
# Keep the working tree configurable for dev2/dev3/dev4, and force the P1
# controller behavior only in the submitted archive.
sed -i \
  "s#    <arg name=\"control_method\" value=\"\$(var control_method)\"/>#    <arg name=\"control_method\" value=\"${control_method}\"/>#" \
  "${staging_dir}/${launch_file}"

# P1 behavior must run regardless of which vehicle receives the submission.
# An empty enabled_ego_vehicle_id disables the P1-only guard in control_cmd_mux.
sed -i \
  's#  <arg name="enabled_ego_vehicle_id" default="P1"/>#  <arg name="enabled_ego_vehicle_id" default=""/>#' \
  "${staging_dir}/${pp_mpc_launch_file}"

# Do not ship interpreter caches from a local build. They can be stale and are
# not part of the submitted source package.
tar czf "${archive}" \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*/.pytest_cache' \
  -C "${staging_dir}" \
  aichallenge_submit

# Fail early if the archive would start a different controller in evaluation.
if ! tar xOf "${archive}" "${launch_file}" | grep -Fq "<arg name=\"control_method\" value=\"${control_method}\"/>"; then
  echo "error: submission archive does not force ${control_method}" >&2
  exit 1
fi

if ! tar xOf "${archive}" "${pp_mpc_launch_file}" | grep -Fq '<arg name="enabled_ego_vehicle_id" default=""/>'; then
  echo "error: submission archive does not disable the P1-only ego-vehicle gate" >&2
  exit 1
fi

echo "created ${archive} (P1 behavior for all vehicles: ${control_method})"
