# test_energy.py (Improved Version)
import time
from pyJoules.energy_meter import measure_energy
from pyJoules.device import DeviceFactory
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.exception import NoSuchDeviceError


def test_component(domain_class, domain_id=0):
    """Tries to measure energy for a single component."""
    component_name = domain_class.__name__
    print(f"--- Testing: {component_name} ---")
    try:
        # Manually create the device to see if it initializes
        device = domain_class(domain_id)

        @measure_energy(domains=[domain_class(domain_id)])
        def dummy_work():
            time.sleep(0.1)

        dummy_work()
        print(f"SUCCESS: Successfully measured energy for {component_name}.\n")
        return True
    except NoSuchDeviceError:
        print(f"FAILURE: Could not find hardware sensors for {component_name}.\n")
        return False
    except Exception as e:
        print(f"FAILURE: An unexpected error occurred for {component_name}: {e}\n")
        return False


if __name__ == "__main__":
    print("Starting hardware diagnostic for pyJoules...\n")

    # Test each component individually
    test_component(RaplPackageDomain)  # CPU
    test_component(RaplDramDomain)  # RAM
    test_component(NvidiaGPUDomain)  # GPU
