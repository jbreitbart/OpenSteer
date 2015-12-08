#include "TestResults.h"
#include "TestReporter.h"

#include "TestDetails.h"


#ifdef _OPENMP
#include <omp.h>
#endif

namespace UnitTest {

TestResults::TestResults(TestReporter* testReporter)
    : m_testReporter(testReporter)
    , m_totalTestCount(0)
    , m_failedTestCount(0)
    , m_failureCount(0)
    , m_currentTestFailed(false)
{
}

void TestResults::OnTestStart(TestDetails const& test)
{
    ++m_totalTestCount;
    m_currentTestFailed = false;
    if (m_testReporter)
        m_testReporter->ReportTestStart(test);
}

void TestResults::OnTestFailure(TestDetails const& test, char const* failure)
{
	// bknafla: As far as I can see this is the only place I need to protect from concurrent access
	//          when calling @c CHECK* for a normal @c TEST.
	#pragma omp critical (unit_test_pp_on_test_failure_critical )
	{
		++m_failureCount;
		if (!m_currentTestFailed)
		{
			++m_failedTestCount;
			m_currentTestFailed = true;
		}

		if (m_testReporter)
			m_testReporter->ReportFailure(test, failure);
	}
}

void TestResults::OnTestFinish(TestDetails const& test, float secondsElapsed)
{
    if (m_testReporter)
        m_testReporter->ReportTestFinish(test, secondsElapsed);
}

int TestResults::GetTotalTestCount() const
{
    return m_totalTestCount;
}

int TestResults::GetFailedTestCount() const
{
    return m_failedTestCount;
}

int TestResults::GetFailureCount() const
{
    return m_failureCount;
}


}
