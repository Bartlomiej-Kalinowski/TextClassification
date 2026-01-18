from vectorizer import TfidfPipeline


def test_vectorize_tfidf(mocker):
    mocker.patch('vectorizer.TfidfPipeline.__init__', return_value=None)
    vect = TfidfPipeline(None, None)
    vect.le = mocker.Mock()
    vect.docs = ['a']
    vect.cat = [0]
    vect.vectorizer = mocker.Mock()
    # TEST SCENARIUSZA 1: transform = True (wywołuje fit_transform)
    vect.vectorize(transform = True)
    vect.vectorizer.fit_transform.assert_called_once_with(vect.docs)
    # TEST SCENARIUSZA 2: transform = False (wywołuje transform)
    vect.vectorize(transform=False)
    vect.vectorizer.transform.assert_called_once_with(vect.docs)



