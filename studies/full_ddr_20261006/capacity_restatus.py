"""One read-only snapshot of the exact recovery lifecycle, never a resubmit."""
import capacity_reoperate
import prepare_status


if __name__ == "__main__":
    capacity_reoperate.bind_context()
    prepare_status.main()
